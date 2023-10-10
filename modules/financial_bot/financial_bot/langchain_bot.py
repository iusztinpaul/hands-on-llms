import logging

from langchain import chains
from langchain.memory import ConversationBufferMemory

from financial_bot import constants
from financial_bot.chains import ContextExtractorChain, FinancialBotQAChain
from financial_bot.embeddings import EmbeddingModelSingleton
from financial_bot.models import build_huggingface_pipeline
from financial_bot.qdrant import build_qdrant_client
from financial_bot.template import get_llm_template

logger = logging.getLogger(__name__)


class FinancialBot:
    def __init__(
        self,
        llm_model_id: str = constants.LLM_MODEL_ID,
        llm_lora_model_id: str = constants.LLM_QLORA_CHECKPOINT,
        debug: bool = constants.DEBUG,
    ):
        self._qdrant_client = build_qdrant_client()
        self._embd_model = EmbeddingModelSingleton()
        self._llm_agent = build_huggingface_pipeline(
            llm_model_id=llm_model_id, llm_lora_model_id=llm_lora_model_id, debug=debug
        )
        self.finbot_chain = self.build_chain()

    def build_chain(self) -> chains.SequentialChain:
        """
        Constructs and returns a financial bot chain.
        This chain is designed to take as input the user description, `about_me` and a `question` and it will
        connect to the VectorDB, searches the financial news that rely on the user's question and injects them into the
        payload that is further passed as a prompt to a financial fine-tuned LLM that will provide answers.

        The chain consists of two primary stages:
        1. Context Extractor: This stage is responsible for embedding the user's question,
        which means converting the textual question into a numerical representation.
        This embedded question is then used to retrieve relevant context from the VectorDB.
        The output of this chain will be a dict payload.

        2. LLM Generator: Once the context is extracted,
        this stage uses it to format a full prompt for the LLM and
        then feed it to the model to get a response that is relevant to the user's question.

        Returns
        -------
        chains.SequentialChain
            The constructed financial bot chain.

        Notes
        -----
        The actual processing flow within the chain can be visualized as:
        [about: str][question: str] > ContextChain >
        [about: str][question:str] + [context: str] > FinancialChain >
        [answer: str]
        """

        logger.info("Building 1/3 - ContextExtractorChain")
        context_retrieval_chain = ContextExtractorChain(
            embedding_model=self._embd_model,
            vector_store=self._qdrant_client,
            vector_collection=constants.VECTOR_DB_OUTPUT_COLLECTION_NAME,
            top_k=constants.VECTOR_DB_SEARCH_TOPK,
        )

        logger.info("Building 2/3 - FinancialBotQAChain")
        llm_generator_chain = FinancialBotQAChain(
            hf_pipeline=self._llm_agent,
            template=get_llm_template(name=constants.TEMPLATE_NAME),
        )

        logger.info("Building 3/3 - Connecting chains into SequentialChain")
        # TODO: Change memory to keep TOP k messages or a summary of the conversation.
        seq_chain = chains.SequentialChain(
            memory=ConversationBufferMemory(
                memory_key="chat_history", input_key="question"
            ),
            chains=[context_retrieval_chain, llm_generator_chain],
            input_variables=["about_me", "question"],
            output_variables=["answer"],
            verbose=True,
        )

        logger.info("Done building SequentialChain.")
        logger.info("Workflow:")
        logger.info(
            """
            [about: str][question: str] > ContextChain > 
            [about: str][question:str] + [context: str] > FinancialChain > 
            [answer: str]
            """
        )

        return seq_chain

    def answer(self, about_me: str, question: str) -> str:
        """
        Given a short description about the user and a question make the LLM
        generate a response.

        Parameters
        ----------
        about_me : str
            Short user description.
        question : str
            User question.

        Returns
        -------
        str
            LLM generated response.
        """

        inputs = {"about_me": about_me, "question": question}
        response = self.finbot_chain.run(inputs)

        return response
