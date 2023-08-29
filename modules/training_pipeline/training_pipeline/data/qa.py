from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from datasets import Dataset
from training_pipeline.constants import Scope
from training_pipeline.data.utils import load_json
from training_pipeline.templates.prompter import get_llm_template


@dataclass(frozen=True)
class DataSample:
    about_me: str = field(repr=False)
    context: str = ""
    question: str = ""
    response: str = ""


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
                    about_me=sample["about_me"],
                    context=sample["context"],
                    question=sample["question"],
                    response=sample["response"],
                )
                for sample in data
            ]
        else:
            return [
                DataSample(
                    about_me=sample["about_me"],
                    context=sample["context"],
                    question=sample["question"],
                )
                for sample in data
            ]

    def to_huggingface(self) -> Dataset:
        """Configures as HF dataset format."""
        data_as_dict = [asdict(sample) for sample in self._raw_data]
        dataset = Dataset.from_list(data_as_dict)
        if self._scope == Scope.TRAINING:
            mapping_func = self._template.format_train
        else:
            mapping_func = self._template.format_infer
        dataset = dataset.map(mapping_func, remove_columns=dataset.column_names)

        return dataset
