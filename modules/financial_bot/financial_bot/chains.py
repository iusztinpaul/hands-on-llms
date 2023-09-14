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
        # Get question text
        question_str = inputs.get("question", None)

        # Embed the question
        embeddings = self.embedding_model(question_str)

        # Search vector store for closest top-k embeddings
        matches = self.vector_store.search(
            query_vector=embeddings, k=self.top_k, collection_name="test_collection"
        )

        # Extract content
        content = matches[0].payload["summary"]

        # Build prompt
        prompt = self.template.infer_raw_template.format(
            user_context=inputs["about_me"], news_context=content, question=question_str
        )

        # Return content
        return {self.output_key: prompt}


class FinancialBotQAChain(Chain):
    top_k: int = 1
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
        response = self.hf_pipeline(inputs[self.input_key])

        # Return content
        return {self.output_key: response}
