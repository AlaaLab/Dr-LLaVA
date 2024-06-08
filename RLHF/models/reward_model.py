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

from argparse import Namespace
import os
from typing import Optional, Dict, Sequence, Union

import einops
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import transformers
from transformers.trainer_utils import EvalPrediction
from transformers.utils.generic import ModelOutput
import pandas as pd
import math
# Set the display options
pd.set_option('display.max_rows', None)  # Replace None with a specific number if you want to limit the rows
pd.set_option('display.max_columns', None)  # Replace None with a specific number if you want to limit the columns
pd.set_option('display.width', None)  # This will try to use the maximum width of your console
pd.set_option('display.max_colwidth', None)  # -1 means unlimited column width, but be cautious as very long text might make the display unwieldy

from peft import PeftModel, LoraModel, LoraConfig

from models.qlora_model import get_accelerate_model

from llava.model import *

import spacy
import pickle  
import numpy as np


def load_reword_model():
    
    rf_model = Rule_Based_Classifier()
    
    
    return rf_model
    

def unpack_dict(
    d: Dict, keys: Sequence[str], return_type: type = tuple
) -> Union[Sequence, Dict]:
    if return_type in (tuple, list):
        return return_type(d[key] for key in keys)
    elif return_type == dict:
        return {key: d[key] for key in keys}
    else:
        raise ValueError(f"Unknown return_type: {return_type}")


def batch_select(input: Tensor, index: Tensor):
    """Select elements from a batched tensor with a batched index tensor.

    Example:
        input = torch.tensor([
            [0, 1, 2],
            [3, 0, 9],
            [6, 7, 8],
        ])
        index = torch.tensor([[0, 1], [1, 0], [0, 0]])
        batch_select(input, index) = tensor([
            [0, 1],
            [0, 3],
            [6, 6]
        ])
    """
    dummy_index = torch.arange(input.size(0), device=input.device).unsqueeze(-1)
    return input[dummy_index, index]


def make_generative_vlm(
    args: Namespace,
    model_name_or_path: str,
    qlora: bool = False,
    checkpoint_dir: Optional[str] = None,
    adapter_name="lora_default",
    is_trainable=True,
    reuse_base_model=False,
    tokenizer=None,
    **kwargs,
):
    if qlora:
        if checkpoint_dir is None or checkpoint_dir in ["scratch", "none"]:
            return get_accelerate_model(args, None, tokenizer=tokenizer)
        else:
            return get_accelerate_model(
                args,
                checkpoint_dir=checkpoint_dir,
                adapter_name=adapter_name,
                is_trainable=is_trainable,
                reuse_base_model=reuse_base_model,
                tokenizer=tokenizer,
            )
    else:
        raise ValueError(f"Unknown model type: {model_name_or_path}")


def get_transformer_hidden_size(model: transformers.PreTrainedModel):
    if isinstance(model, PeftModel):
        return get_transformer_hidden_size(model.base_model)

    if isinstance(model, LoraModel):
        return get_transformer_hidden_size(model.model)

    if isinstance(model, transformers.GPT2LMHeadModel):
        hidden_size_attr_name = "n_embd"
    elif isinstance(model, transformers.OPTForCausalLM):
        hidden_size_attr_name = "word_embed_proj_dim"
    elif isinstance(model, transformers.T5ForConditionalGeneration):
        hidden_size_attr_name = "d_model"
    elif "modelling_RW.RWModel" in str(
        type(model)
    ) or "modelling_RW.RWForCausalLM" in str(type(model)):
        # TODO(zhiqings): Hack to add support for Falcon.
        hidden_size_attr_name = "hidden_size"
    else:
        # Hack to deal with the fact that transformers library changed the LLaMA model name.
        llama_cls = getattr(
            transformers,
            "LLaMAForCausalLM"
            if hasattr(transformers, "LLaMAForCausalLM")
            else "LlamaForCausalLM",
        )
        if isinstance(model, llama_cls) or "LlamaForCausalLM" in str(type(model)):
            hidden_size_attr_name = "hidden_size"
        else:
            raise ValueError(f"Unknown base_model type: {type(model)}")
        from typing import Any, Mapping
    return getattr(model.config, hidden_size_attr_name)


class RewardConfig(transformers.PretrainedConfig):
    model_type = "reward_model"

    # Huggingface doesn't allow non-kwargs for `__init__`.
    def __init__(self, backbone_model_name_or_path=None, **kwargs):
        super(RewardConfig, self).__init__(**kwargs)
        self.backbone_model_name_or_path = backbone_model_name_or_path


class RewardModelOutput(ModelOutput):
    rewards: Tensor = None


class RewardModel(transformers.PreTrainedModel):
    config_class = RewardConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        args: Namespace,
        config: RewardConfig,
        checkpoint_dir: Optional[str] = None,
        adapter_name="lora_default",
        tokenizer=None,
        **kwargs,
    ):
        super(RewardModel, self).__init__(config)
        self.adapter_name = adapter_name
        self.backbone_model = make_generative_vlm(
            args,
            config.backbone_model_name_or_path,
            checkpoint_dir=checkpoint_dir,
            adapter_name=adapter_name,
            tokenizer=tokenizer,
            **kwargs,
        )
        hidden_size = get_transformer_hidden_size(self.backbone_model)
        reward_head = nn.Linear(hidden_size, 1)
        torch.nn.init.zeros_(reward_head.bias)
        device = next(self.backbone_model.parameters()).device
        self.reward_head = reward_head.to(device)

        if checkpoint_dir is not None:
            reward_head_path = os.path.join(checkpoint_dir, "reward_head")
            if os.path.exists(reward_head_path):
                self.reward_head.load_state_dict(
                    torch.load(
                        reward_head_path,
                        map_location="cpu",
                    )
                )
            else:
                print(f"Warning: reward head not found at {reward_head_path}")

        self.reward_head.requires_grad_(kwargs.get("is_trainable", True))

    def forward(
        self, input_ids, attention_mask=None, images=None, return_dict=True, **kwargs
    ):
        # We only compute the rewards and don't compute the logistic regression loss in this function so that it's
        # easier to use for later stages of reranking / RL training.
        self.backbone_model.set_adapter(self.adapter_name)
        self.backbone_model.config.use_cache = False

        # print(input_ids.shape, images.shape, 'images', images.dtype)
        outputs = self.backbone_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            images=images,
            **kwargs,
        )
        last_hidden_state = outputs.hidden_states[-1]
        assert isinstance(last_hidden_state, torch.Tensor), f"{outputs}"
        # last_hidden_state = outputs.last_hidden_state
        # TODO(zhiqings): Hacking to make sure every parameter is used in the backward pass.
        logits = outputs.logits
        last_hidden_state = last_hidden_state + 0.0 * torch.mean(logits)

        last_hidden_state_at_the_end = last_hidden_state[:, -1, :]
        # TODO(lxuechen): Make returning rewards at all positions and last_hidden_state an option.
        # last_hidden_state_at_the_end = last_hidden_state_at_the_end.type_as(
        #     next(self.reward_head.parameters()) # HACK(sheng): error with data parallel
        # )
        last_hidden_state_at_the_end = last_hidden_state_at_the_end.type_as(
            self.reward_head.weight
        )
        # print(last_hidden_state_at_the_end.device, self.reward_head.weight.device, self.reward_head.bias.device)
        rewards = self.reward_head(last_hidden_state_at_the_end).squeeze(-1)
        return RewardModelOutput(rewards=rewards) if return_dict else (rewards,)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, transformers.LlamaModel):
            module.gradient_checkpointing = value

        # TODO(zhiqings): Hack to add support for Falcon.
        if "RWModel" in str(type(module)):
            module.gradient_checkpointing = value


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class RewardModelTrainer(transformers.Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ["mm_projector", "embed_tokens", "embed_in"]
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v

            current_folder = output_dir.split("/")[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(
                    weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                )

        super(RewardModelTrainer, self)._save(output_dir, state_dict)

    def compute_loss(self, model, inputs, return_outputs=False):
        # input_ids, attention_mask each of size (bsz, num_candidates, seq_len).
        # index_0, index_1 each of size (bsz, num_pairs); indexes into input_ids.
        # choice of size (bsz, num_pairs); 1 if index_1's seq is chosen, 0 otherwise.
        input_ids, attention_mask, index_0, index_1, choice, images = unpack_dict(
            inputs,
            keys=(
                "input_ids",
                "attention_mask",
                "index_0",
                "index_1",
                "choice",
                "images",
            ),
        )
        # repeat images to match the number of candidates
        images = images.unsqueeze(1).repeat(1, input_ids.size(1), 1, 1, 1)
        images = einops.rearrange(images, "b n h w c -> (b n) h w c")

        num_candidates, num_pairs = input_ids.size(1), choice.size(1)
        input_ids_flat, attention_mask_flat = tuple(
            einops.rearrange(x, "b c l -> (b c) l") for x in (input_ids, attention_mask)
        )
        outputs = model(
            input_ids=input_ids_flat, attention_mask=attention_mask_flat, images=images
        )
        rewards_flat = outputs.rewards
        rewards = einops.rearrange(
            rewards_flat, "(b c) -> b c", c=num_candidates
        )  # Size: (bsz, num_candidates).

        rewards_0, rewards_1 = tuple(
            batch_select(rewards, index) for index in (index_0, index_1)
        )  # Size: (bsz, num_pairs).
        logits = rewards_1 - rewards_0  # Size: (bsz, num_pairs).
        # Type casting of `choice` is due to amp.autocast context manager.
        loss = F.binary_cross_entropy_with_logits(
            logits, choice.to(logits.dtype), reduction="mean"
        )

        loss = loss + (rewards_1 + rewards_0).mean().abs() * 1e-3

        logged_rewards = torch.stack((rewards_1, rewards_0), dim=-1)
        return (loss, dict(logits=logged_rewards)) if return_outputs else loss


def compute_reward_modeling_metrics(eval_prediction: EvalPrediction) -> Dict:
    # eval_prediction.label_ids is a tuple that matches up with `training_args.label_names`.
    logits = torch.tensor(
        eval_prediction.predictions[..., 0] - eval_prediction.predictions[..., 1]
    ).squeeze(-1)
    labels = torch.tensor(eval_prediction.label_ids[-1]).squeeze(-1)
    predictions = (logits >= 0.0).long()
    accuracy = predictions.eq(labels).float().mean().item()
    label_positive_rate = (labels == 1).float().mean().item()
    average_score = torch.tensor(eval_prediction.predictions).float().mean().item()
    return dict(
        accuracy=accuracy,
        label_positive_rate=label_positive_rate,
        average_score=average_score,
    )


def load_4bit_reward_model_for_inference(
    checkpoint_dir: str,
    vision_tower: str = None,
    lora_modules: list = None,
    image_aspect_ratio: str = "square",
    image_grid_pinpoints: int = None,
    bits: int = 4,
    fp16: bool = False,
    bf16: bool = False,
    double_quant: bool = True,
    quant_type: str = "nf4",
    gradient_checkpointing: bool = False,
    adapter_name="lora_default",
    is_trainable=True,
    reuse_base_model=False,
    trust_remote_code=False,
):
    # Load the model.
    lora_checkpoint_dir = checkpoint_dir
    if os.path.exists(os.path.join(lora_checkpoint_dir, "adapter_model")):
        lora_checkpoint_dir = os.path.join(lora_checkpoint_dir, "adapter_model")
    if os.path.exists(os.path.join(lora_checkpoint_dir, "lora_default")):
        lora_checkpoint_dir = os.path.join(lora_checkpoint_dir, "lora_default")

    lora_config = LoraConfig.from_pretrained(lora_checkpoint_dir)
    config = RewardConfig(
        backbone_model_name_or_path=lora_config.base_model_name_or_path
    )

    args = Namespace(
        model_name_or_path=config.backbone_model_name_or_path,
        vision_tower=vision_tower,
        lora_modules=lora_modules,
        image_aspect_ratio=image_aspect_ratio,
        image_grid_pinpoints=image_grid_pinpoints,
        bits=bits,
        fp16=fp16,
        bf16=bf16,
        double_quant=double_quant,
        quant_type=quant_type,
        trust_remote_code=trust_remote_code,
        full_finetune=False,
        gradient_checkpointing=gradient_checkpointing,
    )

    model = RewardModel(
        args,
        config,
        checkpoint_dir=checkpoint_dir,
        qlora=bits == 4 or bits == 8,
        adapter_name=adapter_name,
        is_trainable=is_trainable,
        reuse_base_model=reuse_base_model,
    )
    return model


def exist_in_sentence(sentence, keywords):
    # check if keywords exist in the sentence
    # sentence: a string
    # keywords: a list of string
    for keyword in keywords:
        if keyword in sentence:
            return True
    return False

def no_exist_in_sentence(sentence, keywords):
    # check if keywords exist in the sentence
    # sentence: a string
    # keywords: a list of string
    for keyword in keywords:
        if keyword in sentence:
            return False
    return True

class Rule_Based_Classifier:
    """
    A classifier that uses rule-based methods to analyze and classify medical diagnosis data.
    """
    def __init__(self):
        self.knowledge = {
            'NORMAL': [True, 'adequate', 'normal', 'no abnormality', 'normal'],
            'AML': [True, 'adequate', 'abnormal', 'myeloblasts', 'AML'],
            'MM': [True, 'adequate', 'abnormal', 'plasma cells', 'MM'],
            'BLOOD': [False, 'blood', 'inadequate', 'inadequate', 'inadequate'],
            'CLOT': [False, 'clot', 'inadequate', 'inadequate', 'inadequate']
        }

    def _get_method(self, index):
        """
        Returns the method associated with the given category index.
        """
        method_dict = {
            1: self._check_quality,
            2: self._image_analysis,
            3: self._pathology_analysis,
            4: self._detailed_abnormality_reasoning,
            5: self._diagnosis
        }
        return method_dict.get(index, None)  # Returns None if index is not found

    # write 5 function to check if each answer is cosistent with the question
    # the 5 question cover 5 aspects
    # Low quality detection; Image overall analysis; Pathology abnormality analysis;Detailed abnormality reasoning; Diagnosis
    def _check_quality(self, sentences):
        sentences = sentences.lower()  #This pathological image segment cannot be adequately utilized for accurate medical diagnosis
        # True, mean it is the good quality
        if exist_in_sentence(sentences, ['effective', 'appropriate', 'suit','apt ','optimal']) and no_exist_in_sentence(sentences, [' not ', ' no ', ' inadequate ',' unsuitable' ]):
            return True
        elif exist_in_sentence(sentences, ['cannot', 'not', 'no', 'inadequate','unsuitable']):
            return False
        else:
            return  'no match'
    
    def _image_analysis(self, sentences):
        sentences = sentences.lower() 
        # there are three conditions: blood, clot and adequate
        if exist_in_sentence(sentences, ['optimal', 'advantageous', 'suitable', 'adequate','well', 'prime']) and \
           exist_in_sentence(sentences, ['close to', 'near', 'close', 'adjacent', 'proximity','vicinity','proximate']):
            return 'adequate'
        
        elif exist_in_sentence(sentences, ['blood', 'rbc','rbcs']):
            return 'blood'
        
        elif no_exist_in_sentence(sentences, ['close to', 'near', 'close', 'adjacent', 'proximity to','vicinity']) and \
             exist_in_sentence(sentences, ['unsuit', 'hinder','less', 'negative', 'adverse']) and \
                exist_in_sentence(sentences, ['particles']):
            return 'clot'
        else:
            return  'no match'

    def _pathology_analysis(self, sentences):
        sentences = sentences.lower() 
        # there are two conditions: normal and abnormal and inadequate
        if exist_in_sentence(sentences, ['malig', 'cancer',' disorder']):
            return 'abnormal'
        elif exist_in_sentence(sentences, ['medical','illne']) and exist_in_sentence(sentences, ['display']):
            return 'abnormal'
        elif exist_in_sentence(sentences, [ ' normal', 'no abnormali']): 
            return 'normal'
        elif exist_in_sentence(sentences, ['inadequate', 'impossible', 'low-quality', 'low quality','unclear ','unsuitable','insufficient']):
            return 'inadequate'
        elif exist_in_sentence(sentences, ['quality']) and exist_in_sentence(sentences, ['low', 'poor', 'insuffici', 'inadequate','subpar']):
            return 'inadequate'
        else:
            return  'no match'

    def _detailed_abnormality_reasoning(self, sentences):
        sentences = sentences.lower() 
        if exist_in_sentence(sentences, ['plasma','plasma']):
            return 'plasma cells'
        elif exist_in_sentence(sentences, ['myeloblast', 'myeloblast']):
            return 'myeloblasts'
        elif exist_in_sentence(sentences, ['quality']) and (exist_in_sentence(sentences, ['low', 'poor', 'insuffici', 'inadequate', 'insufficient']) or \
                                                            exist_in_sentence(sentences, ['not', 'no ', 'devoid', 'free', 'absent'])):
            return 'inadequate'
        elif exist_in_sentence(sentences, ['quality']) and (exist_in_sentence(sentences, ['low', 'poor', 'insuffici', 'inadequate', 'insufficient']) or \
                                                            exist_in_sentence(sentences, ['not', 'no ', 'devoid', 'free', 'absent'])):
            return 'inadequate'
        elif exist_in_sentence(sentences, ['no ', 'not', 'devoid', 'free', 'absent' ,'absence']):
            return 'no abnormality'
        else:
            return  'no match'

    def _diagnosis(self, sentences):
        sentences = sentences.lower() 
        if (exist_in_sentence(sentences, ['blood cancer']) and \
                exist_in_sentence(sentences, ['not', 'no ', 'devoid', 'free', 'absent', 'absence']) and \
                   no_exist_in_sentence(sentences, [' quality'])) or exist_in_sentence(sentences, [' healthy']):
            return 'normal'
        elif exist_in_sentence(sentences, ['multiple myeloma', 'multiple myeloma', 'mm']):
            return 'MM'
        elif exist_in_sentence(sentences, ['acute myeloid leukemia', 'acute myeloid leukemia', 'aml']):
            return 'AML'

        elif exist_in_sentence(sentences, 'quality') and (exist_in_sentence(sentences, ['low', 'poor', 'insuffici', 'inadequate', 'insufficient']) or \
                                                            exist_in_sentence(sentences, ['not', 'no ', 'devoid', 'free', 'absent'])):
            return 'inadequate'
       
        elif exist_in_sentence(sentences, ['blood cancer',]): 
            return 'cancer'
        else:
            return  'no match'
    
    def calculate_reward(self, sentences,  return_dict=True, device =None, ref_answers=None,
    categories=None):
        """
        This function is used to calculate the accumulated reward for a series of sentences
        """
        outcomes = []
        gt = []
        # sort the sentences and ref_answers by categories
        
        
        for sentence, category in zip(sentences, categories):
            method = self._get_method(category)
            if not method:
                raise ValueError(f"No method found for category {category}")
            outcome = method(sentence)
            outcomes.append(outcome)
        

        for ref_answer, category in zip(ref_answers, categories):
            method = self._get_method(category)
            if not method:
                raise ValueError(f"No method found for category {category}")
            gt_result = method(ref_answer)
            gt.append(gt_result)

        correct_bonus = [1 if x == y else -0.5 if x == 'no match' else 0 for x, y in zip(outcomes, gt)]
        align_bonus = []
        for i in range(1,len(categories)):
            res = [[self.knowledge[key][categories[x]-1] for x in [i-1,i]] for key in self.knowledge.keys()]
            if outcomes[i-1: i+1] in res:
                align_bonus.append(1)
            else:
                align_bonus.append(0)
        # length_bonus is calculated if the length of the outcomes is the same as the length of the ref_answers has more than 10 letters difference
        length_bonus = [ - math.abs(len(x)-len(y))/10 if (math.abs(len(x)-len(y))/10)>1 else 0 for x, y in zip(outcomes, gt)]
        # calculate the total bonus
        bonus = sum(correct_bonus) + sum(align_bonus) + sum(length_bonus)
        return RewardModelOutput(rewards=bonus) if return_dict else (None,)
        