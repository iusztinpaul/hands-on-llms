import dotenv
import fire

from financial_bot.langchain_bot import FinancialBot

dotenv.load_dotenv()


def main():
    bot = FinancialBot()
    input_payload = {
        "about_me": "I'm a student and I have some money that I want to invest.",
        "question": "Should I consider investing in stocks from the Tech Sector?",
    }
    response = bot.answer(**input_payload)
    print(response)

    next_question = "What about the Energy Sector?"
    input_payload["question"] = next_question
    response = bot.answer(**input_payload)
    print(response)


if __name__ == "__main__":
    fire.Fire(main)
