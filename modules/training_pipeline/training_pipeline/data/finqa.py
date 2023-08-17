from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset

from training_pipeline.constants import Scope
from training_pipeline.data.utils import load_json


@dataclass
class FinQASample:
    id: str

    pre_text: List[str]
    post_text: List[str]
    table: List[List[str]]

    question: str
    answer: str
    steps: List[Dict[str, str]]
    program: str


@dataclass
class FinQATestingSample:
    id: str

    pre_text: List[str]
    post_text: List[str]
    table: List[List[str]]

    question: str


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

    def load(self, data_path: Path) -> List[FinQASample]:
        data = load_json(data_path)
        if self._max_samples is not None:
            data = data[:self._max_samples]

        return self.deserialize(data)
    
    def deserialize(self, data: List[dict]) -> List[Union[FinQASample, FinQATestingSample]]:
        if self._scope == Scope.TRAINING:
            return [
                FinQASample(
                    id=sample["id"],
                    pre_text=sample["pre_text"],
                    post_text=sample["post_text"],
                    table=sample["table"],
                    question=sample["qa"]["question"],
                    answer=sample["qa"]["answer"],
                    steps=sample["qa"]["steps"],
                    program=sample["qa"]["program"],
                )
                for sample in data
            ]
        else:
            return [
                FinQATestingSample(
                    id=sample["id"],
                    pre_text=sample["pre_text"],
                    post_text=sample["post_text"],
                    table=sample["table"],
                    question=sample["qa"]["question"],
                )
                for sample in data
            ]

    def to_huggingface(self) -> Dataset:
        # TODO: should I add a "eos_token" at the end of the prompt,
        #  as in the following example: sample["text"] = f"{format_dolly(sample)}{tokenizer.eos_token}" ?

        data_as_dict = [asdict(sample) for sample in self._raw_data]
        dataset = Dataset.from_list(data_as_dict)
        if self._scope == Scope.TRAINING:
            mapping_func = self.to_training_prompt
        else:
            mapping_func = self.to_testing_prompt
        dataset = dataset.map(mapping_func, remove_columns=dataset.column_names)

        return dataset

    def to_training_prompt(self, sample: dict) -> str:
        parse_sample = self._parse_sample_training(sample)

        prompt = self.to_testing_prompt(sample)

        answer_prompt = f"### ASSISTANT:\n\
        ### ANSWER: {parse_sample['answer']}\n\
        ### REASONING STEPS:\n \
        {parse_sample['steps']}\n \
        ### PROGRAM compiled from reasoning steps above:\n \
        {parse_sample['program']}\n \
        "

        prompt = f"{prompt['text']}\n\n{answer_prompt}"

        return {
            "text": prompt,
        }

    def to_testing_prompt(self, sample: dict) -> str:
        parsed_sample = self._parse_sample_testing(sample)
        
        system_prompt = f"### SYSTEM: You are a professional financial advisor. Your task is to read a financial report as text and numbers and do the proper math calculations to answer the given question."

        human_prompt = f"### Human:\n \
        ### START_FINANCIAL_REPORT\n \
        ### PRE_TEXT:\n \
        {parsed_sample['pre_text']}\n \
        ### TABLE:\n \
        {parsed_sample['table']}\n \
        ### POST_TEXT:\n \
        {parsed_sample['post_text']}\n \
        ### END_FINANCIAL_REPORT\n \
        ### QUESTION: {parsed_sample['question']} \
        "

        prompt = f"{system_prompt}\n\n{human_prompt}"

        return {
            "text": prompt,
        }

    def _parse_sample_testing(self, sample: Dict[str, Any]) -> Dict[str, str]:
        # TODO: Refactor _parse_sample...() methods.
        pre_text = sample['pre_text']
        pre_text = "\n".join(pre_text)

        table = sample['table']
        table_rows = []
        for table_row in table:
            table_row = " | ".join(table_row)
            table_rows.append(table_row)
        table = "\n".join(table_rows)

        post_text = sample['post_text']
        post_text = "\n".join(post_text)

        return {
            "pre_text": pre_text,
            "table": table,
            "post_text": post_text,
            "question": sample['question'],
        }

    def _parse_sample_training(self, sample: Dict[str, Any]) -> Dict[str, str]:
        parsed_sample = self._parse_sample_testing(sample)
        
        formatted_steps = []
        for i, step in enumerate(sample['steps']):
            formatted_step = f"###STEP {i}: {step['arg1']} {step['op']} {step['arg2']} = {step['res']}"
            formatted_steps.append(formatted_step)
        formatted_steps = "\n".join(formatted_steps)

        return {
            **parsed_sample,
            "answer": sample['answer'],
            "steps": formatted_steps,
            "program": sample['program'],
        }
