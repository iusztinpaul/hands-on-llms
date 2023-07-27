from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset

from common.constants import Scope
from common.data.utils import load_json


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


class FinQADataset:
    def __init__(self, data_path: Path, scope: Scope = Scope.TRAINING):
        self._data_path = data_path
        self._scope = scope

        self._raw_data = self.load(data_path)

    def load(self, data_path: Path) -> List[FinQASample]:
        data = load_json(data_path)

        return self.deserialize(data)
    
    def deserialize(self, data: List[dict]) -> List[dict]:
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

    def to_huggingface(self) -> Dataset:
        data_as_dict = [asdict(sample) for sample in self._raw_data]
        dataset = Dataset.from_list(data_as_dict)
        if self._scope == Scope.TRAINING:
            mapping_func = self.to_training
        else:
            mapping_func = self.to_prompt
        dataset = dataset.map(mapping_func, remove_columns=dataset.column_names)

        return dataset

    def to_training(self, sample: dict) -> str:
        parse_sample = self._parse_sample(sample)

        prompt = self.to_prompt(sample)

        answer_prompt = f"\
        ###Assistant:\n\
        ###ANSWER: {parse_sample['answer']}\n\
        Reasoning steps:\n \
        {parse_sample['steps']}\n \
        Final program based on the reasoning steps:\n \
        {parse_sample['program']}\n \
        "

        prompt = f"{prompt['text']}\n{answer_prompt}"

        return {
            "text": prompt,
        }

    def to_prompt(self, sample: dict) -> str:
        parsed_sample = self._parse_sample(sample)
        
        system_prompt = f"\
        ###System: You are a professional financial advisor. Your task is to read a financial report and answer the given question.\n \
        The financial report you will get will be in the following format:\n \
        ###START_FINANCIAL_REPORT\n \
        ###PRE_TEXT:\n \
        <some text>\n \
        ###TABLE:\n \
        <table with the financial data>\n \
        ###POST_TEXT:\n \
        <some text>\n \
        ###END_FINANCIAL_REPORT\n \
        Your question from the human will be in the following format:\n \
        ###Human:\n\
        ###QUESTION: <question>\n \
        Your answer as an assisant should be in the following format:\n \
        ###Assistant:\n\
        ###ANSWER: <answer>\n \
        You should output all the reasoning steps in the following format:\n \
        ###STEP 0: <arg1> <operation> <arg2> = <result #0> \n \
        ###STEP 1: <arg1> <operation> <arg2> = <result #1>\n \
        ...\
        ###STEP N:<arg1> <operation> <arg2> = <result #N> \n \
        You should also output the final program based on the previus steps as follows:\n \
        ###PROGRAM: <program>\n \
        "

        human_prompt = f"\
        ###Human:\n\
        ###START_FINANCIAL_REPORT\n \
        ###PRE_TEXT:\n \
        {parsed_sample['pre_text']}\n \
        ###TABLE:\n \
        {parsed_sample['table']}\n \
        ###POST_TEXT:\n \
        {parsed_sample['post_text']}\n \
        ###END_FINANCIAL_REPORT\n \
        ###QUESTION: {parsed_sample['question']} \
        "

        prompt = f"{system_prompt}\n{human_prompt}"

        return {
            "text": prompt,
        }

    def _parse_sample(self, sample: Dict[str, Any]) -> Dict[str, str]:
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

        formatted_steps = []
        for i, step in enumerate(sample['steps']):
            formatted_step = f"###STEP {i}: {step['arg1']} {step['op']} {step['arg2']} = {step['res']}"
            formatted_steps.append(formatted_step)
        formatted_steps = "\n".join(formatted_steps)

        return {
            "pre_text": pre_text,
            "table": table,
            "post_text": post_text,
            "question": sample['question'],
            "answer": sample['answer'],
            "steps": formatted_steps,
            "program": sample['program'],
        }
