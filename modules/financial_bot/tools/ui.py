import logging
from threading import Thread
from typing import List

import gradio as gr

from financial_bot import load_bot

logger = logging.getLogger(__name__)

bot = load_bot()


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
    textbox=gr.Textbox(placeholder="Ask me a financial question", label="Financial question", container=False, scale=7),
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
