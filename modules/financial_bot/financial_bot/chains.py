from typing import Any, Dict, List

import qdrant_client
from financial_bot.embeddings import EmbeddingModelSingleton
from financial_bot.template import PromptTemplate
from langchain.chains.base import Chain
from langchain.llms import HuggingFacePipeline


class ContextExtractorChain(Chain):
    """
    Encode the question, search the vector store for top-k articles and return
    context news from documents collection of Alpaca news.
    """

    top_k: int = 1
    embedding_model: EmbeddingModelSingleton
    vector_store: qdrant_client.QdrantClient
    vector_collection: str
    output_key: str = "payload"

    @property
    def input_keys(self) -> List[str]:
        return ["about_me", "question", "context"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: handle that None, without the need to enter chain
        about_key, quest_key, contx_key = self.input_keys
        question_str = inputs.get(quest_key, None)

        # TODO: maybe async embed?
        embeddings = self.embedding_model(question_str)

        # TODO: get rid of hardcoded collection_name, specify 1 top_k or adjust multiple context insertions
        matches = self.vector_store.search(
            query_vector=embeddings,
            k=self.top_k,
            collection_name=self.vector_collection,
        )

        content = ""
        for match in matches:
            content += match.payload["summary"] + "\n"

        payload = {
            about_key: inputs[about_key],
            quest_key: inputs[quest_key],
            contx_key: content,
        }

        return {self.output_key: payload}


class FinancialBotQAChain(Chain):
    """This custom chain handles LLM generation upon given prompt"""

    hf_pipeline: HuggingFacePipeline
    template: PromptTemplate
    output_key: str = "response"

    @property
    def input_keys(self) -> List[str]:
        return ["payload"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: use .get and treat default value?
        about_me = inputs["about_me"]
        question = inputs["question"]
        context = inputs["context"]

        prompt = self.template.infer_raw_template.format(
            user_context=about_me, news_context=context, question=question
        )
        response = self.hf_pipeline(prompt)

        return {self.output_key: response}
