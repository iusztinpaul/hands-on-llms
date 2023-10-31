from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs

from training_pipeline.constants import Scope
from training_pipeline.data.utils import load_json
from training_pipeline.prompt_templates.prompter import get_llm_template


@dataclass(frozen=True)
class DataSample:
    user_context: str = field(repr=False)
    news_context: str = ""
    chat_history: str = ""
    question: str = ""
    answer: str = ""


class FinanceDataset:
    def __init__(
        self,
        data_path: Path,
        scope: Scope = Scope.TRAINING,
        template: str = "falcon",
        max_samples: Optional[int] = None,
    ):
        self._data_path = data_path
        self._scope = scope
        self._max_samples = max_samples
        self._template = get_llm_template(template)
        self._raw_data = self.load(data_path)

    def load(self, data_path: Path) -> List[DataSample]:
        data = load_json(data_path)
        if self._max_samples is not None:
            data = data[: self._max_samples]

        return self.deserialize(data)

    def deserialize(self, data: List[dict]) -> List[DataSample]:
        if self._scope == Scope.TRAINING:
            return [
                DataSample(
                    user_context=sample["about_me"],
                    news_context=sample["context"],
                    chat_history=sample.get("chat_history", ""),
                    question=sample["question"],
                    answer=sample["response"],
                )
                for sample in data
            ]
        else:
            return [
                DataSample(
                    user_context=sample["about_me"],
                    news_context=sample["context"],
                    chat_history=sample.get("chat_history", ""),
                    question=sample["question"],
                )
                for sample in data
            ]

    def to_huggingface(self) -> Dataset:
        """Configures as HF dataset format."""
        data_as_dict = [asdict(sample) for sample in self._raw_data]
        dataset = Dataset.from_list(data_as_dict)
        if self._scope == Scope.TRAINING:
            template_mapping_func = self._template.format_train
        else:
            template_mapping_func = self._template.format_infer
        
        dataset = dataset.map(self.clean)
        dataset = dataset.map(template_mapping_func, remove_columns=dataset.column_names)

        return dataset

    def clean(self, samples: Dict[str, str]) -> Dict[str, str]:
        for key, sample in samples.items():
            cleaned_sample = clean_extra_whitespace(sample)
            cleaned_sample = group_broken_paragraphs(cleaned_sample)
            cleaned_sample = cleaned_sample.strip("\n_\\ ")
            
            samples[key] = cleaned_sample

        return samples
