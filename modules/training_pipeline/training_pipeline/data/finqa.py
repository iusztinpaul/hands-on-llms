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
            data = data[: self._max_samples]

        return self.deserialize(data)
    
    @property
    def question_template(self) -> str:
        return \
        """
        ### SYSTEM: You are a professional financial advisor. Your task is to read a financial report 
        as text and numbers and do the proper math calculations to answer the given question.

                
        ### Human:
        ### START_FINANCIAL_REPORT
        ### PRE_TEXT:
        {pre_text}

        #### TABLE:
        {table}

        ### POST_TEXT:
        {post_text}
        #### END_FINANCIAL_REPORT

        ### QUESTION: 
        {question}
        """
    
    @property
    def answer_template(self) -> str:
        return \
        """
        ### ASSISTANT:
        ### ANSWER:
        {answer}

        ### REASONING STEPS:
        {reasoning_steps}

        ### PROGRAM compiled from reasoning steps above:
        {program}
        """
    
    @property
    def question_and_answer_template(self) -> str:
        return f"{self.question_template}\n\n{self.answer_template}"

    def deserialize(
        self, data: List[dict]
    ) -> List[Union[FinQASample, FinQATestingSample]]:
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
            mapping_func = self.to_question_and_answer_prompt
        else:
            mapping_func = self.to_question_prompt
        dataset = dataset.map(mapping_func, remove_columns=dataset.column_names)

        return dataset

    def to_question_prompt(self, sample: dict) -> str:
        variables = self._get_question_variables(sample)

        return {
            "text": self.question_template.format(**variables),
            "template": self.question_template,
            "variables": variables
        }
    
    def _get_question_variables(self, sample: Dict[str, Any]) -> Dict[str, str]:
        pre_text = sample["pre_text"]
        pre_text = "\n".join(pre_text)

        table = sample["table"]
        table_rows = []
        for table_row in table:
            table_row = " | ".join(table_row)
            table_rows.append(table_row)
        table = "\n".join(table_rows)

        post_text = sample["post_text"]
        post_text = "\n".join(post_text)

        return {
            "pre_text": pre_text,
            "table": table,
            "post_text": post_text,
            "question": sample["question"],
        }

    
    def to_question_and_answer_prompt(self, sample: dict) -> str:
        variables = self._get_question_and_answer_variables(sample)

        return {
            "text": self.question_and_answer_template.format(**variables),
            "template": self.question_and_answer_template,
            "variables": variables
        }
    
    def _get_question_and_answer_variables(self, sample: Dict[str, Any]) -> Dict[str, str]:
        parsed_sample = self._get_question_variables(sample)

        formatted_steps = []
        for i, step in enumerate(sample["steps"]):
            formatted_step = f"###STEP {i}: {step['arg1']} {step['op']} {step['arg2']} = {step['res']}"
            formatted_steps.append(formatted_step)
        formatted_steps = "\n".join(formatted_steps)

        return {
            **parsed_sample,
            "answer": sample["answer"],
            "reasoning_steps": formatted_steps,
            "program": sample["program"],
        }
