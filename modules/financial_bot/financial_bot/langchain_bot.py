import logging
import os

import constants
import models
import qdrant
from embeddings import EmbeddingModelSingleton
from langchain.chains import RetrievalQA
from langchain.chains.base import Chain
from langchain.llms import HuggingFacePipeline
from template import get_llm_template

logger = logging.getLogger(__name__)


class FinancialBot:
    def __init__(self, template_name: str = constants.TEMPLATE_NAME):
        self._hf_pipeline = models.build_huggingface_pipeline()
        self._qdrant = qdrant.build_qdrant_client(
            url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"]
        )
        self._embeddings_model = EmbeddingModelSingleton()
        self._prompt_template = get_llm_template(template_name)
        self.qa_chain = self.build_context_chain()

    def build_context_chain(self):
        retriever = self._qdrant.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=self._hf_pipeline,
            retriever=retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self._prompt_template.infer_raw_template},
        )
        return qa_chain

    def answer(self, about_me: str, question: str):
        try:
            result = self.qa_chain({"query": question})
        except Exception as e:
            pass


if __name__ == "__main__":
    pass
