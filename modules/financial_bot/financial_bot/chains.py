import time
from typing import Any, Dict, List, Optional

import qdrant_client
from langchain import chains
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.llms import HuggingFacePipeline
from unstructured.cleaners.core import (
    clean,
    clean_extra_whitespace,
    clean_non_ascii_chars,
    group_broken_paragraphs,
    replace_unicode_quotes,
)

from financial_bot.embeddings import EmbeddingModelSingleton
from financial_bot.template import PromptTemplate


class StatelessMemorySequentialChain(chains.SequentialChain):
    history_input_key: str = "to_load_history"

    def _call(self, inputs: Dict[str, str], **kwargs) -> Dict[str, str]:
        """Override _call to load history before calling the chain."""

        to_load_history = inputs[self.history_input_key]
        for (
            human,
            ai,
        ) in to_load_history:
            self.memory.save_context(
                inputs={self.memory.input_key: human},
                outputs={self.memory.output_key: ai},
            )
        memory_values = self.memory.load_memory_variables({})
        inputs.update(memory_values)

        del inputs[self.history_input_key]

        return super()._call(inputs, **kwargs)

    def prep_outputs(
        self,
        inputs: Dict[str, str],
        outputs: Dict[str, str],
        return_only_outputs: bool = False,
    ) -> Dict[str, str]:
        """Override prep_outputs to clear the internal memory after each call."""

        results = super().prep_outputs(inputs, outputs, return_only_outputs)

        # Clear the internal memory.
        self.memory.clear()
        if self.memory.memory_key in results:
            results[self.memory.memory_key] = ""

        return results


class ContextExtractorChain(Chain):
    """
    Encode the question, search the vector store for top-k articles and return
    context news from documents collection of Alpaca news.
    """

    top_k: int = 1
    embedding_model: EmbeddingModelSingleton
    vector_store: qdrant_client.QdrantClient
    vector_collection: str

    @property
    def input_keys(self) -> List[str]:
        return ["about_me", "question"]

    @property
    def output_keys(self) -> List[str]:
        return ["context"]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        _, quest_key = self.input_keys
        question_str = inputs[quest_key]

        cleaned_question = self.clean(question_str)
        # TODO: Chunk the question in 'max_input_length' chunks,
        # pass them through the model and average the embeddings.
        cleaned_question = cleaned_question[: self.embedding_model.max_input_length]
        embeddings = self.embedding_model(cleaned_question)

        # TODO: Using the metadata filter the news from the latest week (or other timeline).
        matches = self.vector_store.search(
            query_vector=embeddings,
            k=self.top_k,
            collection_name=self.vector_collection,
        )

        context = ""
        for match in matches:
            context += match.payload["summary"] + "\n"

        return {
            "context": context,
        }

    def clean(self, question: str) -> str:
        question = clean(question)
        question = replace_unicode_quotes(question)
        question = clean_non_ascii_chars(question)

        return question


class FinancialBotQAChain(Chain):
    """This custom chain handles LLM generation upon given prompt"""

    hf_pipeline: HuggingFacePipeline
    template: PromptTemplate

    @property
    def input_keys(self) -> List[str]:
        return ["context"]

    @property
    def output_keys(self) -> List[str]:
        return ["answer"]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        inputs = self.clean(inputs)
        prompt = self.template.format_infer(
            {
                "user_context": inputs["about_me"],
                "news_context": inputs["context"],
                "chat_history": inputs["chat_history"],
                "question": inputs["question"],
            }
        )

        start_time = time.time()
        response = self.hf_pipeline(prompt["prompt"])
        end_time = time.time()
        duration_milliseconds = (end_time - start_time) * 1000

        if run_manager:
            run_manager.on_chain_end(
                outputs={
                    "answer": response,
                },
                # TODO: Count tokens instead of using len().
                metadata={
                    "prompt": prompt["prompt"],
                    "prompt_template_variables": prompt["payload"],
                    "prompt_template": self.template.infer_raw_template,
                    "usage.prompt_tokens": len(prompt["prompt"]),
                    "usage.total_tokens": len(prompt["prompt"]) + len(response),
                    "usage.actual_new_tokens": len(response),
                    "duration_milliseconds": duration_milliseconds,
                },
            )

        return {"answer": response}

    def clean(self, inputs: Dict[str, str]) -> Dict[str, str]:
        for key, input in inputs.items():
            cleaned_input = clean_extra_whitespace(input)
            cleaned_input = group_broken_paragraphs(cleaned_input)

            inputs[key] = cleaned_input

        return inputs
