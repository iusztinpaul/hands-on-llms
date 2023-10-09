import dotenv
import fire

dotenv.load_dotenv()

from financial_bot.langchain_bot import FinancialBot


def main():
    bot = FinancialBot()
    input_payload = {
        "about_me": "I'm a student and I have some money that I want to invest.",
        "question": "Should I consider investing in stocks from the Tech Sector?",
        "context": ""
    }

    response = bot.answer(**input_payload)
    print(response)


if __name__ == "__main__":
    fire.Fire(main)
