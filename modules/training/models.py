from typing import Optional
import torch

from peft import LoraConfig, PeftModel, PeftConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def build_qlora_model(
    model_id: str = "tiiuae/falcon-7b", peft_model_id: Optional[str] = None
):
    """
    Function that builds a QLoRA LLM model based on the given HuggingFace name:
        1.   Create and prepare the bitsandbytes configuration for QLoRa's quantization
        2.   Download, load, and quantize on-the-fly Falcon-7b
        3.   Create and prepare the LoRa configuration
        4.   Load and configuration Falcon-7B's tokenizer
    """

    # TODO: Double-check resources and improve this func.

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        load_in_4bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

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

    return model, tokenizer, lora_config


def prompt(model, tokenizer, input_text: str, max_new_tokens: int = 40, device: str = "cuda:0"):
    # TODO: Rewrite this function using the huggingface pipeline class.
    # Example:
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

    tokenizer.return_token_type_ids = False

    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    # TODO: How can I get rid of token_type_ids in a cleaner way?
    del inputs["token_type_ids"]

    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    output = outputs[0] # The input to the model is a batch of size 1, so the output is also a batch of size 1.
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return output
