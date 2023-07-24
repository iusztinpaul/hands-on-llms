from typing import Optional
import torch

from peft import LoraConfig, PeftModel, PeftConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def build_qlora_model(model_id: str = "tiiuae/falcon-7b-instruct", peft_model_id: Optional[str] = None):
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

    peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["query_key_value"],
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    if peft_model_id:
        model = PeftModel.from_pretrained(model, peft_model_id)

    return model, tokenizer, peft_config
