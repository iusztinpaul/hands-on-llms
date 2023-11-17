"""
This script defines a PromptTemplate class that assists in generating 
conversation/prompt templates. The script facilitates formatting prompts 
for inference and training by combining various context elements and user inputs.
"""


import dataclasses
from typing import Dict, List, Union


@dataclasses.dataclass
class PromptTemplate:
    """A class that manages prompt templates"""

    name: str
    system_template: str = "{system_message}"
    context_template: str = "{user_context}\n{news_context}"
    chat_history_template: str = "{chat_history}"
    question_template: str = "{question}"
    answer_template: str = "{answer}"
    system_message: str = ""
    sep: str = "\n"
    eos: str = ""

    @property
    def input_variables(self) -> List[str]:
        """Returns a list of input variables for the prompt template"""

        return ["user_context", "news_context", "chat_history", "question", "answer"]

    @property
    def train_raw_template(self):
        """Returns the training prompt template format"""

        system = self.system_template.format(system_message=self.system_message)
        context = f"{self.sep}{self.context_template}"
        chat_history = f"{self.sep}{self.chat_history_template}"
        question = f"{self.sep}{self.question_template}"
        answer = f"{self.sep}{self.answer_template}"

        return f"{system}{context}{chat_history}{question}{answer}{self.eos}"

    @property
    def infer_raw_template(self):
        """Returns the inference prompt template format"""

        system = self.system_template.format(system_message=self.system_message)
        context = f"{self.sep}{self.context_template}"
        chat_history = f"{self.sep}{self.chat_history_template}"
        question = f"{self.sep}{self.question_template}"

        return f"{system}{context}{chat_history}{question}{self.eos}"

    def format_train(self, sample: Dict[str, str]) -> Dict[str, Union[str, Dict]]:
        """Formats the data sample to a training sample"""

        prompt = self.train_raw_template.format(
            user_context=sample["user_context"],
            news_context=sample["news_context"],
            chat_history=sample.get("chat_history", ""),
            question=sample["question"],
            answer=sample["answer"],
        )

        return {"prompt": prompt, "payload": sample}

    def format_infer(self, sample: Dict[str, str]) -> Dict[str, Union[str, Dict]]:
        """Formats the data sample to a testing sample"""

        prompt = self.infer_raw_template.format(
            user_context=sample["user_context"],
            news_context=sample["news_context"],
            chat_history=sample.get("chat_history", ""),
            question=sample["question"],
        )
        return {"prompt": prompt, "payload": sample}


# Global Templates registry
templates: Dict[str, PromptTemplate] = {}


def register_llm_template(template: PromptTemplate):
    """Register a new template to the global templates registry"""

    templates[template.name] = template


def get_llm_template(name: str) -> PromptTemplate:
    """Returns the template assigned to the given name"""

    return templates[name]


##### Register Templates #####
# - FALCON (spec: https://huggingface.co/tiiuae/falcon-7b/blob/main/tokenizer.json)
register_llm_template(
    PromptTemplate(
        name="falcon",
        system_template=">>INTRODUCTION<< {system_message}",
        system_message="You are a helpful assistant, with financial expertise.",
        context_template=">>DOMAIN<< {user_context}\n{news_context}",
        chat_history_template=">>SUMMARY<< {chat_history}",
        question_template=">>QUESTION<< {question}",
        answer_template=">>ANSWER<< {answer}",
        sep="\n",
        eos="<|endoftext|>",
    )
)
