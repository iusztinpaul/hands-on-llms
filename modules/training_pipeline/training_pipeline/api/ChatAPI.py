import logging
from typing import Tuple

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig

from training_pipeline import constants, models


logger = logging.getLogger(__name__)


class ChatAPI:
    def __init__(
        self,
        peft_model_id: str,
        model_id: str = constants.MODEL_ID_DEFAULT,
        max_new_tokens: int = 50,
        device: str = "cuda:0",
    ):
        self._peft_model_id = peft_model_id
        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._device = device

        self._model, self._tokenizer, self._peft_config = self.load_model()

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]:
        logger.info(f"Loading model using {self._model_id=} and {self._peft_model_id=}")

        model, tokenizer, peft_config = models.build_qlora_model(
            model_id=self._model_id,
            peft_model_id=self._peft_model_id,
            gradient_checkpointing=False
            )
        
        return model, tokenizer, peft_config
    
    def ask(self, question: str) -> str:
        answer = models.prompt(
            model=self._model,
            tokenizer=self._tokenizer,
            input_text=question,
            device=self._device,
        )

        return answer
