from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset
from training_pipeline.constants import Scope
from training_pipeline.data.utils import load_json


@dataclass(frozen=True)
class DataSample:
    about_me: str = field(repr=False)
    context: str = ""
    question: str = ""
    response: str = ""


class FinQADataset:
    def __init__(
        self,
        data_path: Path,
        scope: Scope = Scope.TRAINING,
        max_samples: Optional[int] = None,
    ):
        self._data_path = data_path
        self._scope = scope
        self._max_samples = max_samples

        self._raw_data = self.load(data_path)

    def load(self, data_path: Path) -> List[DataSample]:
        data = load_json(data_path)
        if self._max_samples is not None:
            data = data[: self._max_samples]

        return self.deserialize(data)

    @property
    def question_template(self) -> str:
        """
        Formats to Falcon compatible template, using `added_tokens`:
        - INTRODUCTION : specifies agent behaviour
        - COMMENT : used for context and user description
        - QUESTION: formats user question for the agent.
        spec: https://huggingface.co/tiiuae/falcon-7b/raw/main/tokenizer.json
        """
        template = """
        >>INTRODUCTION<<
        You are an expert in the stock and crypto markets.
        
        >>COMMENT<<
        {ABOUT_ME}
        
        >>COMMENT<<
        {CONTEXT}

        >>QUESTION<<
        User: {QUESTION}
        """
        return template

    @property
    def answer_template(self) -> str:
        """
        Formats to Falcon compatible template, using `added_tokens`:
        - ANSWER : specifies agent answer format
        spec: https://huggingface.co/tiiuae/falcon-7b/raw/main/tokenizer.json
        """
        template = """
        >>ANSWER<<
        Assistant: {ANSWER}
        """
        return template

    @property
    def question_and_answer_template(self) -> str:
        return f"{self.question_template}\n\n{self.answer_template}"

    def deserialize(self, data: List[dict]) -> List[DataSample]:
        if self._scope == Scope.TRAINING:
            return [
                DataSample(
                    about_me=sample["about_me"],
                    context=sample["context"],
                    question=sample["question"],
                    response=sample["response"],
                )
                for sample in data
            ]

    def to_huggingface(self) -> Dataset:
        """Configures as HF dataset format."""
        data_as_dict = [asdict(sample) for sample in self._raw_data]
        dataset = Dataset.from_list(data_as_dict)
        if self._scope == Scope.TRAINING:
            mapping_func = self.to_qa_prompt
        else:
            mapping_func = self.to_q_prompt
        dataset = dataset.map(mapping_func, remove_columns=dataset.column_names)

        return dataset

    def to_q_prompt(self, sample: dict) -> str:
        """Formats data sample without response field."""
        formatted_prompt = self.question_template.format(
            ABOUT_ME=sample["about_me"],
            CONTEXT=sample["context"],
            QUESTION=sample["question"],
        )
        # spec: https://github.com/cmp-nct/ggllm.cpp/discussions/36#discussioncomment-6315713
        formatted_prompt += "<|endoftext|>"

        return formatted_prompt

    def to_qa_prompt(self, sample: dict) -> str:
        """Formats sample dict as training-ready."""
        qa_formatted_prompt = self.question_and_answer_template.format(
            ABOUT_ME=sample["about_me"],
            CONTEXT=sample["context"],
            QUESTION=sample["question"],
            ANSWER=sample["response"],
        )

        # spec: https://github.com/cmp-nct/ggllm.cpp/discussions/36#discussioncomment-6315713
        qa_formatted_prompt += "<|endoftext|>"

        return qa_formatted_prompt
