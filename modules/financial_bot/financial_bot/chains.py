from typing import Any, Dict, List

import qdrant_client
from langchain import chains
from langchain.chains.base import Chain
from langchain.llms import HuggingFacePipeline

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

        embeddings = self.embedding_model(question_str)

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

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.template.format_infer(
            {
                "user_context": inputs["about_me"],
                "news_context": inputs["question"],
                "chat_history": inputs["chat_history"],
                "question": inputs.get("context"),
            }
        )["prompt"]
        response = self.hf_pipeline(prompt)

        return {"answer": response}
