# Copyright 2023 The LLaVA-RLHF Team
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
import sys

sys.path.append(
    "/wynton/protected/group/ibrahim/harry/LLaVA_checkpoints/LLaVA-RLHF/llava_setup/LLaVA/"
)
from argparse import Namespace
from typing import Optional
from os.path import join, exists

import torch
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    PeftModel,
    PeftModelForCausalLM,
)
from peft.tuners.lora import LoraLayer

from llava.model import LlavaLlamaForCausalLM

REGISTERED_BASE_MODELS = {}


def find_all_linear_names(
    args: Namespace,
    model: torch.nn.Module,
):
    cls = (
        bnb.nn.Linear4bit
        if args.bits == 4
        else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def get_accelerate_model(
    args: Namespace,
    checkpoint_dir: Optional[str] = None,
    adapter_name="lora_default",
    is_trainable=True,
    reuse_base_model=False,
    tokenizer=None,
):
    global REGISTERED_BASE_MODELS

    compute_dtype = (
        torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )

    if checkpoint_dir is not None:
        if exists(join(checkpoint_dir, "adapter_model")):
            checkpoint_dir = join(checkpoint_dir, "adapter_model")

        if exists(join(checkpoint_dir, "lora_default")):
            checkpoint_dir = join(checkpoint_dir, "lora_default")

    if args.model_name_or_path in REGISTERED_BASE_MODELS and reuse_base_model:
        config = {
            "load_in_4bit": args.bits == 4,
            "load_in_8bit": args.bits == 8,
            "llm_int8_threshold": 6.0,
            "llm_int8_has_fp16_weight": False,
            "bnb_4bit_compute_dtype": compute_dtype,
            "bnb_4bit_use_double_quant": args.double_quant,
            "bnb_4bit_quant_type": args.quant_type,
        }

        registered_model, registered_config = REGISTERED_BASE_MODELS[
            args.model_name_or_path
        ]
        if registered_config == config and not args.full_finetune:
            print(f"loading registered model {args.model_name_or_path}...")
            model = registered_model

            model.load_adapter(
                checkpoint_dir,
                adapter_name=adapter_name,
                is_trainable=is_trainable,
            )
            return model
        else:
            raise ValueError(
                f"Model {args.model_name_or_path} is already registered with a different config."
                f"{registered_config} != {config}"
            )

    current_device = torch.cuda.current_device()
    if args.full_finetune:
        assert args.bits in [16, 32]

    print(f"loading base model {args.model_name_or_path}...")

    model = LlavaLlamaForCausalLM.from_pretrained(
        args.model_name_or_path,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map={"": current_device},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
            llm_int8_skip_modules=["mm_projector", "lm_head"],
        ),
        torch_dtype=(
            torch.float32
            if args.fp16
            else (torch.bfloat16 if args.bf16 else torch.float32)
        ),
        trust_remote_code=args.trust_remote_code,
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print(
                "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
            )
            print("=" * 80)

    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)

    model.config.torch_dtype = (
        torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )

    if not args.full_finetune:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if not args.full_finetune:
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")

            model = PeftModel.from_pretrained(
                model,
                checkpoint_dir,
                adapter_name=adapter_name,
                is_trainable=is_trainable,
            )

            if args.model_name_or_path not in REGISTERED_BASE_MODELS:
                config = {
                    "load_in_4bit": args.bits == 4,
                    "load_in_8bit": args.bits == 8,
                    "llm_int8_threshold": 6.0,
                    "llm_int8_has_fp16_weight": False,
                    "bnb_4bit_compute_dtype": compute_dtype,
                    "bnb_4bit_use_double_quant": args.double_quant,
                    "bnb_4bit_quant_type": args.quant_type,
                }
                REGISTERED_BASE_MODELS[args.model_name_or_path] = (model, config)

        else:
            # print(f'adding LoRA modules...')
            modules = args.lora_modules or find_all_linear_names(args, model)
            print("adding LoRa modules: ", modules)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config, adapter_name=adapter_name)

    # TODO(zhiqings): this has to be done after get_peft_model
    if args.vision_tower is not None:
        model.config.image_aspect_ratio = args.image_aspect_ratio
        model.config.image_grid_pinpoints = args.image_grid_pinpoints

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device="cuda", dtype=torch.bfloat16)
        vision_tower.requires_grad_(False)

        mm_projector = model.get_model().mm_projector
        mm_projector.to(device="cuda", dtype=torch.bfloat16)
        mm_projector.requires_grad_(False)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model


def load_4bit_model_for_inference(
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
    if checkpoint_dir is not None:
        if exists(join(checkpoint_dir, "adapter_model")):
            checkpoint_dir = join(checkpoint_dir, "adapter_model")

        if exists(join(checkpoint_dir, "lora_default")):
            checkpoint_dir = join(checkpoint_dir, "lora_default")

    config = LoraConfig.from_pretrained(checkpoint_dir)
    base_model_name_or_path = config.base_model_name_or_path

    args = Namespace(
        model_name_or_path=base_model_name_or_path,
        vision_tower=vision_tower,
        lora_modules=lora_modules,
        image_aspect_ratio=image_aspect_ratio,
        image_grid_pinpoints=image_grid_pinpoints,
        bits=bits,
        fp16=fp16,
        bf16=bf16,
        double_quant=double_quant,
        quant_type=quant_type,
        gradient_checkpointing=gradient_checkpointing,
        trust_remote_code=trust_remote_code,
        full_finetune=False,
    )

    model = get_accelerate_model(
        args,
        checkpoint_dir=checkpoint_dir,
        adapter_name=adapter_name,
        is_trainable=is_trainable,
        reuse_base_model=reuse_base_model,
    )
    return model


def get_peft_model(model, peft_config, adapter_name="default"):
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]): Model to be wrapped.
        peft_config ([`PeftConfig`]): Configuration object containing the parameters of the Peft model.
    """
    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
    return PeftModelForCausalLM(model, peft_config, adapter_name=adapter_name)
