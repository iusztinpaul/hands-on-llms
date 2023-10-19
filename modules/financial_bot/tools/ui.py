import logging
from pathlib import Path
from threading import Thread
from typing import List

import gradio as gr

logger = logging.getLogger(__name__)


# === Load Bot ===


def load_bot(
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = "./model_cache",
    embedding_model_device: str = "cuda:0",
    debug: bool = False,
):
    """
    Load the financial assistant bot in production or development mode based on the `debug` flag

    production: the embedding model runs on GPU and the fine-tuned LLM is used.
    dev: the embedding model runs on CPU and the fine-tuned LLM is mocked.
    """

    from financial_bot import initialize

    # Be sure to initialize the environment variables before importing any other modules.
    initialize(logging_config_path=logging_config_path, env_file_path=env_file_path)

    from financial_bot import utils
    from financial_bot.langchain_bot import FinancialBot

    logger.info("#" * 100)
    utils.log_available_gpu_memory()
    utils.log_available_ram()
    logger.info("#" * 100)

    bot = FinancialBot(
        model_cache_dir=Path(model_cache_dir) if model_cache_dir else None,
        embedding_model_device=embedding_model_device,
        debug=debug,
    )

    return bot


bot = load_bot()


# === Gradio Interface ===


def predict(message: str, history: List[List[str]], about_me: str):
    generate_kwargs = {
        "about_me": about_me,
        "question": message,
    }

    t = Thread(target=bot.answer, kwargs=generate_kwargs)
    t.start()

    for partial_answer in bot.stream_answer():
        yield partial_answer


demo = gr.ChatInterface(
    predict,
    textbox=gr.Textbox(
        placeholder="Ask me a financial question",
        label="Financial question",
        container=False,
        scale=7,
    ),
    additional_inputs=[
        gr.Textbox(
            "I am a student and I have some money that I want to invest.",
            label="About me",
        )
    ],
    title="Your Personal Financial Assistant",
    description="Ask me any financial or crypto market questions, and I will do my best to answer them.",
    theme="soft",
    examples=[
        [
            "What's your opinion on investing in startup companies?",
            "I am a 30 year old graphic designer. I want to invest in something with potential for high returns.",
        ],
        [
            "What's your opinion on investing in AI-related companies?",
            "I'm a 25 year old entrepreneur interested in emerging technologies. \
             I'm willing to take calculated risks for potential high returns.",
        ],
        [
            "Do you think advancements in gene therapy are impacting biotech company valuations?",
            "I'm a 31 year old scientist. I'm curious about the potential of biotech investments.",
        ],
    ],
    cache_examples=False,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
)


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
