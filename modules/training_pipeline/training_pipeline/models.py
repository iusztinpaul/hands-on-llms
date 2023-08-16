from typing import Optional, Tuple

import transformers
import torch

from peft import LoraConfig, PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def build_qlora_model(
    model_id: str = "tiiuae/falcon-7b-instruct",
    peft_model_id: Optional[str] = None,
    gradient_checkpointing: bool = True,
    cache_dir: str = "../model_cache",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]:
    """
    Function that builds a QLoRA LLM model based on the given HuggingFace name:
        1.   Create and prepare the bitsandbytes configuration for QLoRa's quantization
        2.   Download, load, and quantize on-the-fly Falcon-7b
        3.   Create and prepare the LoRa configuration
        4.   Load and configuration Falcon-7B's tokenizer
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # TODO: For multi-GPU: max_memory = {i: '46000MB' for i in range(torch.cuda.device_count())}
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision="main",
        quantization_config=bnb_config,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    # TODO: Should we also enable kbit training? Check out what it does.
    # from peft import prepare_model_for_kbit_training
    # model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    if peft_model_id:
        lora_config = LoraConfig.from_pretrained(peft_model_id)
        assert (
            lora_config.base_model_name_or_path == model_id
        ), f"Lora Model trained on different base model than the one requested: {lora_config.base_model_name_or_path} != {model_id}"

        # model = get_peft_model(model, lora_config)  # TODO: Why this is not working?
        model = PeftModel.from_pretrained(model, peft_model_id)
    else:
        lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["query_key_value"],
        )

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = (
            False  # Gradient checkpointing is not compatible with caching.
        )
    else:
        model.gradient_checkpointing_disable()
        model.config.use_cache = True  # It is good practice to enable caching when using the model for inference.

    return model, tokenizer, lora_config


def prompt(
    model, tokenizer, input_text: str, max_new_tokens: int = 40, device: str = "cuda:0"
):
    # TODO: Rewrite this function using the huggingface pipeline class.
    # Example: https://huggingface.co/tiiuae/falcon-7b#how-to-get-started-with-the-model
    # pipeline = pipeline(
    #     "text-generation",
    #     model=model_4bit,
    #     tokenizer=tokenizer,
    #     use_cache=True,
    #     device_map="auto",
    #     max_length=296,
    #     do_sample=True,
    #     top_k=10,
    #     num_return_sequences=1,
    #     eos_token_id=tokenizer.eos_token_id,
    #     pad_token_id=tokenizer.eos_token_id,
    # )
    # TODO: Should I add a pytorch with.inference_mode() or torch.no_grad() context manager? when running the inference or is done by default by hugingface?

    # tokenizer.return_token_type_ids = False

    # inputs = tokenizer(input_text, return_tensors="pt").to(device)
    # # TODO: How can I get rid of token_type_ids in a cleaner way?
    # del inputs["token_type_ids"]

    # outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # output = outputs[
    #     0
    # ]  # The input to the model is a batch of size 1, so the output is also a batch of size 1.
    # output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    sequences = pipeline(
        input_text,
        max_length=max_new_tokens,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    return sequences
