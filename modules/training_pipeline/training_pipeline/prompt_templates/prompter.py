"""
This script defines a PrompterTemplate class that assists in generating 
conversation/prompt templates. The script facilitates formatting prompts 
for inference and training by combining various context elements and user inputs.

"""


import dataclasses
from typing import Dict, List, Tuple


@dataclasses.dataclass
class PrompterTemplate:
    name: str
    agent_behaviour: str = """
    >>INTRODUCTION<<
    {system_message}
    """
    roles: List[str] = ("USER", "ASSISTANT")

    def __format_base(self, user_context: str, news_context: str) -> str:
        """
        Formats to Falcon compatible template, using `added_tokens`:
        - INTRODUCTION : specifies agent behaviour
        - COMMENT : used for context and user description
        spec: https://huggingface.co/tiiuae/falcon-7b/raw/main/tokenizer.json
        """
        base_template = f"""
        >>COMMENT<<
        {user_context}
        {news_context}
        """
        return base_template

    def format_infer(self, sample: dict) -> Tuple[str, Dict[str, str]]:
        """
        - QUESTION: formats user question for the agent.
        """
        base_template = self.__format_base(
            user_context=sample["about_me"], news_context=sample["context"]
        )
        question = f"""
        >>QUESTION<<
        {sample["question"]}
        """

        return {"prompt": base_template + question, "payload": sample}

    def format_train(self, sample: dict) -> Tuple[str, Dict[str, str]]:
        """
        - QUESTION: formats user question for the agent.
        - ANSWER: formats user format for the agent.
        """
        base_template = self.__format_base(
            user_context=sample["about_me"], news_context=sample["context"]
        )
        question = f"""
        >>QUESTION<<
        {sample["question"]}
        """

        answer = f"""
         >>ANSWER<<
        {sample["response"]}
        """

        return {"prompt": base_template + question + answer, "payload": sample}

    @property
    def raw_template(self):
        tmp = (
            self.agent_behaviour
            + """
        >>COMMENT<<
        {user_context}
        {news_context}
        >>QUESTION<<
        User: {question}
        >>ANSWER<<
        Answer: {answer}
        """
        )
        return tmp


templates: Dict[str, PrompterTemplate] = {}


def register_llm_template(template: PrompterTemplate):
    """Register a new conversation template."""
    templates[template.name] = template


def get_llm_template(name: str) -> PrompterTemplate:
    """Get a conversation template."""
    return templates[name]


##### Register Templates #####
register_llm_template(
    PrompterTemplate(
        name="falcon",
        agent_behaviour="You are an expert in the stock and crypto markets.",
        roles=("User", "Assistant"),
    )
)
