import dotenv
import fire

dotenv.load_dotenv()

from financial_bot import chains
from financial_bot.embeddings import EmbeddingModelSingleton
from financial_bot.models import build_huggingface_pipeline
from financial_bot.qdrant import build_qdrant_client
from langchain.chains import SequentialChain


def main():
    qdrant = build_qdrant_client()
    embeddings_model = EmbeddingModelSingleton()

    quest_retrieval_chain = chains.PreparePromptChain(
        embedding_model=embeddings_model, vector_store=qdrant
    )

    llm_model = build_huggingface_pipeline()

    llm_chain = chains.FinancialBotQAChain(hf_pipeline=llm_model)
    seq_chain = SequentialChain(
        chains=[quest_retrieval_chain, llm_chain],
        input_variables=["about_me", "question"],
        output_variables=["response"],
        verbose=True,
    )

    response = seq_chain.run(
        {
            "about_me": "I'm a 27 years old IT programmer.",
            "question": "Is it a good time to invest in Microsoft stock?",
        }
    )
    print(response)


if __name__ == "__main__":
    fire.Fire(main)
