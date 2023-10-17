import logging
from pathlib import Path
from threading import Thread

import gradio as gr
import torch
from transformers import (
    StoppingCriteria,
)

logger = logging.getLogger(__name__)


def load_bot(
    env_file_path: str = ".env",
    logging_config_path: str = "logging.yaml",
    model_cache_dir: str = "./model_cache",
    embedding_model_device: str = "cuda:0",
    debug: bool = False,
):
    """Load the Financial Assistant Bot in production mode: the embedding model runs on GPU and the LLM is used."""

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


class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [29, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def predict(message, history, about_me):
    generate_kwargs = {
        "about_me": about_me,
        "question": message,
    }

    t = Thread(target=bot.answer, kwargs=generate_kwargs)
    t.start()

    partial_message = ""
    for new_token in bot._streamer:
        if new_token != "<":
            partial_message += new_token
            yield partial_message


demo = gr.ChatInterface(
    predict,
    additional_inputs=[
        gr.Textbox(
            "I am a student and I have some money that I want to invest.",
            label="About me",
        )
    ],
    title="Your Personal Financial Assistant",
    description="Ask me any question about the financial market and I will try to answer it.",
    theme="soft",
    examples=[
        [
            "Should I consider investing in stocks from the Tech Sector?",
            "I am a student and I have some money that I want to invest.",
        ],
        [
            "Should I invest in ETFs or single picked stocks?",
            "I am a long term investor looking for a safe investment.",
        ],
    ],
    cache_examples=False,
    retry_btn=None,
    undo_btn=None,
    clear_btn="Clear",
    )


if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
