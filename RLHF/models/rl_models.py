# Copyright 2023 The LLaVA-RLHF Team
# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model classes that are shared across different algorithms.

WARNING:
    Do not tamper with the state_dict function for any of these classes.
    If you tamper, make sure the keys are the same, otherwise FSDP will get confused.
"""

import abc
import logging
from typing import Dict, Optional
import random
import torch
import transformers
import torch.nn.functional as F
from torch import Tensor, nn

from data_utils.common_utils import right_pad, compute_logprobs
from models.reward_model import get_transformer_hidden_size


logger = logging.getLogger(__name__)


class Policy(nn.Module, abc.ABC):
    def __init__(
        self,
        args,
        base_model: transformers.PreTrainedModel,
        base_tokenizer: transformers.PreTrainedTokenizer,
        adapter_name: Optional[str] = None,
    ):
        super().__init__()
        self.args = args
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.adapter_name = adapter_name

    @abc.abstractmethod
    def forward(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        responses: Tensor,
        images: Optional[Tensor] = None,
        reward_images: Optional[Tensor] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        images: Optional[Tensor] = None,
        temperature: Optional[float] = None,
        num_return_sequences=1,
    ) -> Dict[str, Tensor]:
        assert not self.training, "Policy must be in eval model for generation."
        return self._post_respond(
            self._respond(
                queries,
                query_attn_masks,
                images,
                temperature,
                num_return_sequences,
            )
        )

    @abc.abstractmethod
    def _respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        images: Optional[Tensor] = None,
        temperature: Optional[float] = None,
        num_return_sequences=1,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def _post_respond(self, respond_outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return respond_outputs


class AutoregressivePolicy(Policy):
    def forward(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        responses: Tensor,
        AnswerQuestionMASK: Tensor,
        images: Optional[Tensor] = None,
        reward_images: Optional[Tensor] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        # TODO(lxuechen): Refactor attention mask. Here query_attn_masks overrides padding-based attention mask.

        if self.adapter_name is not None:
            self.base_model.set_adapter(self.adapter_name)
        self.base_model.config.use_cache = False

        if temperature is None:
            temperature = self.args.temperature
        # print(queries.size())
        # print(responses.size())

        # get First Q
        _queries, _query_attn_masks, _images = get_prompt_image_first_question(
            queries, query_attn_masks, images
        )

        input_ids = torch.cat([_queries, responses], dim=1)
        attention_mask = input_ids.ne(self.base_tokenizer.pad_token_id)
        attention_mask[:, : _queries.size(1)] = _query_attn_masks

        # Fix position id issues and ensure consistency with `respond` for GPT and OPT.
        inputs = self.base_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            use_cache=False,
        )
        outputs = self.base_model(**inputs, output_hidden_states=True)
        original_logits = outputs.logits[:, -self.args.response_len - 1 : -1]
        logits = original_logits / temperature
        labels = input_ids[:, -self.args.response_len :]
        labels[AnswerQuestionMASK == self.base_tokenizer.pad_token_id] = (
            self.base_tokenizer.pad_token_id
        )

        logprobs = compute_logprobs(
            logits, labels, ignore_index=self.base_tokenizer.pad_token_id
        )

        entropies = -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1)
        last_hidden_state = outputs.hidden_states[-1][
            :, -self.args.response_len - 1 : -1
        ]
        return dict(
            original_logits=original_logits,
            logits=logits,
            logprobs=logprobs,
            entropies=entropies,
            last_hidden_state=last_hidden_state,
        )

    def _respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        images: Optional[Tensor] = None,
        temperature: Optional[float] = None,
        num_return_sequences=1,
    ) -> Dict[str, Tensor]:
        if self.adapter_name is not None:
            self.base_model.set_adapter(self.adapter_name)
        self.base_model.config.use_cache = True
        self.base_model.config.cache_shape = (
            queries.shape[-1]
            + self.args.response_len
            + self.base_model.get_vision_tower().num_patches,
        )

        if temperature is None:
            temperature = self.args.temperature

        # queries, query_attn_masks, images = get_different_response(queries, query_attn_masks, images)

        _queries, _query_attn_masks, _images = get_prompt_image_first_question(
            queries, query_attn_masks, images
        )

        sequences = self.base_model.generate(
            inputs=_queries,
            images=_images,
            attention_mask=_query_attn_masks,
            do_sample=True,
            max_new_tokens=self.args.response_len,
            pad_token_id=self.base_tokenizer.pad_token_id,
            suppress_tokens=(
                [self.base_tokenizer.eos_token_id]
                if self.args.suppress_eos_at_generation
                else None
            ),
            top_p=1.0,
            top_k=0,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            synced_gpus=True,
        )
        responses = sequences[:, _queries.size(1) :]

        texts = self.base_tokenizer.batch_decode(
            responses,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        encoded_input = self.base_tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to("cuda")
        answers_tokens = encoded_input["input_ids"]
        answers_attention_mask = encoded_input["attention_mask"]
        together = _queries.clone()
        together_attention = _query_attn_masks.clone()

        question_answers = torch.ones(answers_tokens.size()) * 147
        random.seed()
        for i in range(2, 5):#CHANG TO 5 as we have 4 questions
            # hack the system here because 
            # 1. the first question is always the same
            # 2. there are 5 questions in total.
            # 3. if you design have a different rounds of conversation, you need to change the number of rounds here
            s_queries, s_query_attn_masks = get_follow_up_questions( #, s_images
                queries, query_attn_masks, images, _order=i
            )

            together = torch.cat([together, answers_tokens, s_queries], dim=1)

            together_attention = torch.cat(
                [together_attention, answers_attention_mask, s_query_attn_masks], dim=1
            )
            
            print(i)
            print('answer')
            print(answers_tokens.shape)
            print('answer_attention')
            print(answers_attention_mask.shape)
            print('query')
            print(s_queries.shape)
            print('query attention')
            print(s_query_attn_masks.shape)
            
            print('together')
            print(together.shape)
            print('together attention')
            print(together_attention.shape)
            print('-----------------')

            sequences = self.base_model.generate(
                inputs=together,
                images=_images,
                attention_mask=together_attention,
                do_sample=True,
                max_new_tokens=100,  # Hack here because almost all the answers are less than even 30 tokens
                pad_token_id=self.base_tokenizer.pad_token_id,
                suppress_tokens=(
                    [self.base_tokenizer.eos_token_id]
                    if self.args.suppress_eos_at_generation
                    else None
                ),
                top_p=1.0,
                top_k=0,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                synced_gpus=True,
            )
            _responses = sequences[:, together.size(1) :]

            texts = self.base_tokenizer.batch_decode(
                _responses,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            encoded_input = self.base_tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to("cuda")
            answers_tokens = encoded_input["input_ids"]
            answers_attention_mask = encoded_input["attention_mask"]

            question_answers = torch.cat(
                [
                    question_answers,
                    torch.ones(s_queries.size()),
                    torch.ones(answers_tokens.size()) * 147,
                ],
                dim=1,
            )

        responses = sequences[:, _queries.size(1) :]

        question_answers[question_answers == 1] = self.base_tokenizer.pad_token_id

        question_answers[question_answers == 147] = 1

        responses = right_pad(
            responses,
            target_size=(sequences.size(0), self.args.response_len),
            value=self.base_tokenizer.pad_token_id,
        )

        question_answers = right_pad(
            question_answers,
            target_size=(sequences.size(0), self.args.response_len),
            value=self.base_tokenizer.pad_token_id,
        )

        return dict(
            responses=responses,
            AnswerQuestionMASK=question_answers,
            num_QA=4, #Before 5
        )  # Size (bsz * num_return_sequences, response_len).


class Value(nn.Module, abc.ABC):
    def __init__(
        self,
        args,
        base_model: transformers.PreTrainedModel,
        base_tokenizer: transformers.PreTrainedTokenizer,
        adapter_name: Optional[str] = None,
    ):
        super().__init__()
        self.args = args
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        hidden_size = get_transformer_hidden_size(base_model)
        value_head = torch.nn.Linear(hidden_size, 1)
        value_head.weight.data.zero_()
        value_head.bias.data.zero_()
        self.value_head = value_head.to(next(base_model.parameters()).device)
        self.adapter_name = adapter_name

    @abc.abstractmethod
    def forward(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        responses: Tensor,
        images: Optional[Tensor] = None,
        reward_images: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError


class AutoregressiveValue(Value):
    def forward(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        responses: Tensor,
        AnswerQuestionMASK: Tensor,
        images: Optional[Tensor] = None,
        reward_images: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if self.adapter_name is not None:
            self.base_model.set_adapter(self.adapter_name)
        self.base_model.config.use_cache = False

        _queries, _query_attn_masks, _images = get_prompt_image_first_question(
            queries, query_attn_masks, images
        )
        sequences = torch.cat([_queries, responses], dim=1)
        sequence_attn_masks = sequences.ne(self.base_tokenizer.pad_token_id)

        inputs = self.base_model.prepare_inputs_for_generation(
            input_ids=sequences,
            attention_mask=sequence_attn_masks,
            images=reward_images,
            use_cache=False,
        )
        outputs = self.base_model(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
        )

        last_hidden_state = outputs.hidden_states[-1]
        assert isinstance(last_hidden_state, torch.Tensor), f"{outputs}"
        logits = outputs.logits
        last_hidden_state = last_hidden_state + 0.0 * torch.mean(logits)
        last_hidden_state = last_hidden_state[:, -responses.size(1) - 1 : -1]

        last_hidden_state = last_hidden_state.type_as(
            next(self.value_head.parameters())
        )
        values = self.value_head(last_hidden_state).squeeze(-1)
        return dict(values=values)


class ActorCritic(nn.Module):
    def __init__(self, policy: Policy, value_model: Value):
        super(ActorCritic, self).__init__()
        self.policy = policy
        self.value_model = value_model

    def forward(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        responses: Tensor,
        AnswerQuestionMASK: Tensor,
        images: Optional[Tensor] = None,
        reward_images: Optional[Tensor] = None,
        temperature: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        # Assume the policy and value model share the same tokenizer.
        if mode is None:
            o1 = self.policy(
                queries,
                query_attn_masks,
                responses,
                AnswerQuestionMASK,
                images,
                reward_images,
                temperature,
            )
            o2 = self.value_model(
                queries,
                query_attn_masks,
                responses,
                AnswerQuestionMASK,
                images,
                reward_images,
            )

        elif mode == "policy":
            o1 = self.policy(
                queries,
                query_attn_masks,
                responses,
                AnswerQuestionMASK,
                images,
                reward_images,
                temperature,
            )
            # Add dummy loss to make sure every parameter is used in the backward pass.
            o2 = {
                "dummy_loss": 0.0
                * torch.sum(
                    torch.stack(
                        [
                            torch.mean(value)
                            for key, value in self.named_parameters()
                            if "lora_value" in key
                        ]
                    )
                )
            }
        elif mode == "value":
            o2 = self.value_model(
                queries,
                query_attn_masks,
                responses,
                AnswerQuestionMASK,
                images,
                reward_images,
            )
            # Add dummy loss to make sure every parameter is used in the backward pass.
            o1 = {
                "dummy_loss": 0.0
                * torch.sum(
                    torch.stack(
                        [
                            torch.mean(value)
                            for key, value in self.named_parameters()
                            if "lora_policy" in key
                        ]
                    )
                )
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return {**o1, **o2}

    def respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        images: Optional[Tensor] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        return self.policy.respond(
            queries=queries,
            query_attn_masks=query_attn_masks,
            images=images,
            temperature=temperature,
        )


def make_policy_with_base_model(
    args,
    base_model: transformers.PreTrainedModel,
    base_tokenizer: transformers.PreTrainedTokenizer,
    adapter_name: Optional[str] = "default",
) -> Policy:
    if base_model.config.is_encoder_decoder:
        raise NotImplementedError
    else:
        return AutoregressivePolicy(
            args, base_model, base_tokenizer, adapter_name=adapter_name
        )


def make_value_with_base_model(
    args,
    base_model: transformers.PreTrainedModel,
    base_tokenizer: transformers.PreTrainedTokenizer,
    adapter_name: Optional[str] = "default",
) -> Value:
    if base_model.config.is_encoder_decoder:
        raise NotImplementedError
    else:
        return AutoregressiveValue(
            args, base_model, base_tokenizer, adapter_name=adapter_name
        )


def pad_and_stack_tensors(tensor_list, target_size=256):
    """
    Pads and stacks a list of 1D tensors to a specified size.

    Args:
    tensor_list (list of torch.Tensor): List of 1D tensors.
    target_size (int): The size to pad the tensors to.

    Returns:
    torch.Tensor: A stacked tensor of shape (len(tensor_list), target_size).
    """
    padded_tensors = []

    for tensor in tensor_list:

        # Calculate the padding size

        padding_size = max(0, target_size - tensor.size(1))

        # Pad the tensor and add it to the list
        padded_tensor = F.pad(tensor, (padding_size, 0))
        padded_tensors.append(padded_tensor)

    # Stack all the padded tensors
    stacked_tensor = torch.stack(padded_tensors, dim=0)
    stacked_tensor = stacked_tensor.squeeze(1)

    return stacked_tensor


def get_prompt_image_first_question(
    queries,
    queries_attention_mask,
    images,
):
    """
    Because the format in the conversation setup is 
    prompt + {image} + first question + {RESPONSE} + second question + {RESPONSE} + third question + {RESPONSE} + fourth question + {RESPONSE} + fifth question + {RESPONSE}
    # THE FIRST RETRIVED MUST BE PROMPT AND IMAGE AND FIRST QUESTION
    """
    assert (
        queries.shape[0] == queries_attention_mask.shape[0] == images.shape[0]
    ), "the dimension does not match with each other"
    final_queries_list = []
    final_attention_list = []
    final_image_stacks = []
    for _index in range(queries.shape[0]):
        split_indices = (queries[_index] == 29871).nonzero(as_tuple=True)[0]
        start_idx = 0
        p = 0

        promt_and_image = []

        attention_promt_and_image = []

        prepare_question_attention_list = []
        prepare_question_list = []

        for idx in split_indices:

            if p <= 1:

                promt_and_image.append(queries[_index][start_idx : idx + 1])
                attention_promt_and_image.append(
                    queries_attention_mask[_index][start_idx : idx + 1]
                )
                # Slice the tensor from the current start index to the current 29871 index
                p += 1
                start_idx = idx + 1

                if p == 2:
                    prompt = torch.cat(promt_and_image)
                    attention_prompt = torch.cat(attention_promt_and_image)
            else:
                prepare_question_list.append(
                    torch.cat([prompt, queries[_index][start_idx:idx]])
                )
                prepare_question_attention_list.append(
                    torch.cat(
                        [
                            attention_prompt,
                            queries_attention_mask[_index][start_idx:idx],
                        ]
                    )
                )
                start_idx = idx + 1
                p = p + 1
                break

        # prepare_question_list.append(torch.cat([prompt,queries[_index][start_idx:]]))
        # prepare_question_attention_list.append(torch.cat([attention_prompt,queries_attention_mask[_index][start_idx:]]))

        length = prepare_question_list[0].size(0)
        prepare_question = prepare_question_list[0].view(1, length)
        prepare_question_attention = prepare_question_attention_list[0].view(1, length)

        unsqueezed_tensor = images[_index].unsqueeze(0)
        _images = unsqueezed_tensor.repeat(len(prepare_question_list), 1, 1, 1)
        final_image_stacks.append(_images)
        final_queries_list.append(prepare_question)
        final_attention_list.append(prepare_question_attention)

    images = torch.cat(final_image_stacks, axis=0)
    queries = pad_and_stack_tensors(
        final_queries_list, target_size=max([x.size(1) for x in final_queries_list])
    )
    attentions = pad_and_stack_tensors(
        final_attention_list, target_size=max([x.size(1) for x in final_attention_list])
    )
    return queries, attentions, images




def get_follow_up_questions(queries, queries_attention_mask, images, _order=None):
    assert (
        queries.shape[0] == queries_attention_mask.shape[0] == images.shape[0]
    ), "The dimensions do not match each other"

    final_queries_list, final_attention_list, = [], []

    for _index in range(queries.shape[0]):
        split_indices = (queries[_index] == 29871).nonzero(as_tuple=True)[0]
        start_idx = 0
        p = 0

        prompt_and_image, attention_prompt_and_image = [], []
        prepare_question_list, prepare_question_attention_list = [], []
        final_case = False

        for idx in split_indices:
            if p <= 1:
                prompt_and_image.append(queries[_index][start_idx:idx + 1])
                attention_prompt_and_image.append(
                    queries_attention_mask[_index][start_idx:idx + 1]
                )
                p += 1
                start_idx = idx + 1
            else:
                if p == _order + 1:
                    prepare_question_list.append(queries[_index][start_idx:idx])
                    prepare_question_attention_list.append(
                        queries_attention_mask[_index][start_idx:idx]
                    )
                    final_case = False
                    break
                start_idx = idx + 1
                p += 1
                final_case = True

        if final_case:
            assert len(prepare_question_list) == 0, "Should be empty"
            prepare_question_list.append(queries[_index][start_idx:])
            prepare_question_attention_list.append(
                queries_attention_mask[_index][start_idx:]
            )

        length = prepare_question_list[0].size(0)
        prepare_question = prepare_question_list[0].view(1, length)
        prepare_question_attention = prepare_question_attention_list[0].view(1, length)


       
        final_queries_list.append(prepare_question)
        final_attention_list.append(prepare_question_attention)

    
    queries = pad_and_stack_tensors(
        final_queries_list, target_size=max(x.size(1) for x in final_queries_list)
    )

    attentions = pad_and_stack_tensors(
        final_attention_list, target_size=max(x.size(1) for x in final_attention_list)
    )

    return queries, attentions


def get_first_response(queries, queries_attention_mask, images, ):
    assert queries.shape[0] == queries_attention_mask.shape[0] == images.shape[0], 'the dimension does not match with each other'
    final_queries_list = []
    final_attention_list = []
    final_image_stacks = []
    for _index  in range(queries.shape[0]):
        split_indices = (queries[_index] == 29871).nonzero(as_tuple=True)[0]     
        start_idx = 0
        p = 0
        
        promt_and_image = []
        question_list = []

        attention_promt_and_image = []
        attention_question_list = []


        prepare_question_attention_list = []
        prepare_question_list = []
        
        for idx in split_indices:
            
            if p <=1:
                
                    
                promt_and_image.append(queries[_index][start_idx:idx+1])
                attention_promt_and_image.append(queries_attention_mask[_index][start_idx:idx+1])
                        # Slice the tensor from the current start index to the current 29871 index
                p+=1
                start_idx = idx + 1

                if p ==2:
                    prompt = torch.cat(promt_and_image)
                    attention_prompt = torch.cat(attention_promt_and_image)
            else:
                prepare_question_list.append(torch.cat([prompt, queries[_index][start_idx:idx]]))
                prepare_question_attention_list.append(torch.cat([attention_prompt,queries_attention_mask[_index][start_idx:idx]]))
                start_idx = idx + 1
                p = p+1
                break
        
        #prepare_question_list.append(torch.cat([prompt,queries[_index][start_idx:]]))
        #prepare_question_attention_list.append(torch.cat([attention_prompt,queries_attention_mask[_index][start_idx:]]))
        
        
        length = prepare_question_list[0].size(0)
        prepare_question = prepare_question_list[0].view(1, length)
        prepare_question_attention = prepare_question_attention_list[0].view(1, length)

        
        unsqueezed_tensor = images[_index].unsqueeze(0)
        _images = unsqueezed_tensor.repeat(len(prepare_question_list), 1, 1, 1)
        final_image_stacks.append(_images)
        final_queries_list.append(prepare_question)
        final_attention_list.append(prepare_question_attention)

    

    images = torch.cat(final_image_stacks, axis = 0) 
    #print('Largest tensor size')
    #print(max([x.size(1) for x in final_queries_list ]))
    queries = pad_and_stack_tensors(final_queries_list,target_size=max([x.size(1) for x in final_queries_list]))
    
    attentions = pad_and_stack_tensors(final_attention_list,target_size=max([x.size(1) for x in final_attention_list]))
    
    return queries, attentions, images

def get_single_response(queries, queries_attention_mask, images,_order =None):
    assert queries.shape[0] == queries_attention_mask.shape[0] == images.shape[0], 'the dimension does not match with each other'
    final_queries_list = []
    final_attention_list = []
    final_image_stacks = []
    for _index  in range(queries.shape[0]):
        split_indices = (queries[_index] == 29871).nonzero(as_tuple=True)[0]     
        start_idx = 0
        p = 0
        
        promt_and_image = []
        question_list = []

        attention_promt_and_image = []
        attention_question_list = []




        prepare_question_attention_list = []
        prepare_question_list = []
        final_case = False
        for idx in split_indices:
            
            if p <=1:
                
                    
                promt_and_image.append(queries[_index][start_idx:idx+1])
                attention_promt_and_image.append(queries_attention_mask[_index][start_idx:idx+1])
                        # Slice the tensor from the current start index to the current 29871 index
                p+=1
                start_idx = idx + 1

                if p ==2:
                    prompt = torch.cat(promt_and_image)
                    attention_prompt = torch.cat(attention_promt_and_image)
            else:
                if p ==_order+1:
                    prepare_question_list.append(queries[_index][start_idx:idx])
                    prepare_question_attention_list.append(queries_attention_mask[_index][start_idx:idx])
                    final_case = False
                    break
                start_idx = idx + 1
                p = p+1
                final_case = True
            
                
        if final_case:
            assert len(prepare_question_list) ==0, 'shall be empty'
            prepare_question_list.append(queries[_index][start_idx:])
            prepare_question_attention_list.append(queries_attention_mask[_index][start_idx:])
        
        
        length = prepare_question_list[0].size(0)
        prepare_question = prepare_question_list[0].view(1, length)
        prepare_question_attention = prepare_question_attention_list[0].view(1, length)

       

        
        unsqueezed_tensor = images[_index].unsqueeze(0)
        _images = unsqueezed_tensor.repeat(len(prepare_question_list), 1, 1, 1)
        final_image_stacks.append(_images)
        final_queries_list.append(prepare_question)
        final_attention_list.append(prepare_question_attention)

    #images = torch.cat(final_image_stacks, axis = 0) 
    #queries = torch.cat(final_queries_list,  axis = 0)
    #attentions = torch.cat(final_attention_list, axis = 0)
    images = torch.cat(final_image_stacks, axis = 0) 
    queries = pad_and_stack_tensors(final_queries_list,target_size=max([x.size(1) for x in final_queries_list]))
    
    attentions = pad_and_stack_tensors(final_attention_list,target_size=max([x.size(1) for x in final_attention_list]))

    return queries, attentions, images 
