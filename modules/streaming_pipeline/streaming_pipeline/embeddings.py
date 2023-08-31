from typing import Union
import numpy as np
from streaming_pipeline import constants
from transformers import AutoModel, AutoTokenizer

from streaming_pipeline.base import SingletonMeta



class EmbeddingModelSingleton(metaclass=SingletonMeta):
    def __init__(
        self,
        model_id: str = constants.EMBEDDING_MODEL_ID,
        max_input_length: int = constants.EMBEDDING_MODEL_MAX_INPUT_LENGTH,
        device: str = constants.EMBEDDING_MODEL_DEVICE,
    ):
        self._device = device
        self._max_input_length = max_input_length

        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id).to(self._device)

    @property
    def max_input_length(self) -> int:
        return self._max_input_length

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    def __call__(self, input_text: str, to_list: bool = True) -> Union[np.ndarray, list]:
        tokenized_text = self._tokenizer(
            input_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self._max_input_length,
        ).to(self._device)
        result = self._model(**tokenized_text)

        embeddings = result.last_hidden_state[:, 0, :].cpu().detach().numpy()
        if to_list:
            embeddings = embeddings.flatten().tolist()

        return embeddings
