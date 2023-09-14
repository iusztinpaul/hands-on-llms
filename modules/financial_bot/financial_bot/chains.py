from typing import Any, Dict, List, Optional

import financial_bot.template as template
import qdrant_client
from financial_bot.embeddings import EmbeddingModelSingleton
from langchain.chains.base import Chain
from langchain.llms import HuggingFacePipeline


class PreparePromptChain(Chain):
    """
    Encode the question, search the vector store for top-k articles and return
    context news from documents collection of Alpaca news.
    """

    top_k: int = 1
    embedding_model: EmbeddingModelSingleton
    vector_store: qdrant_client.QdrantClient
    template = template.get_llm_template("falcon")
    output_key: str = "prompt"

    @property
    def input_keys(self) -> List[str]:
        return ["about_me", "question"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: handle that None, without the need to enter chain
        question_str = inputs.get("question", None)

        # TODO: maybe async embed?
        embeddings = self.embedding_model(question_str)

        # TODO: get rid of hardcoded collection_name, specify 1 top_k or adjust multiple context insertions
        matches = self.vector_store.search(
            query_vector=embeddings, k=self.top_k, collection_name="test_collection"
        )

        content = matches[0].payload["summary"]

        prompt = self.template.infer_raw_template.format(
            user_context=inputs["about_me"], news_context=content, question=question_str
        )

        # TODO: this input_keys,output_keys looks like a factory?
        return {self.output_key: prompt}


class FinancialBotQAChain(Chain):
    """This custom chain handles LLM generation upon given prompt"""

    hf_pipeline: HuggingFacePipeline
    input_key: str = "prompt"
    output_key: str = "response"

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: this chain looks too simple, maybe make a single chain, or extend this one to
        # be more complex
        response = self.hf_pipeline(inputs[self.input_key])

        return {self.output_key: response}
