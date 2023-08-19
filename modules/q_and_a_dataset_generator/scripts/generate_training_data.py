import os
import openai
from typing import Dict
from tqdm import tqdm

from src.logger import get_console_logger
from src.paths import DATA_DIR

logger = get_console_logger()

PROMPT_TEMPLATE = """
You are an expert in the stock and crypto markets. I will give you some information about myself and you will provide me with good investment advice.

# ABOUT ME
{ABOUT_ME}

# CONTEXT
{CONTEXT}

Please provide concrete advice in less than 100 tokens, and justify your answer based on the news provided in the context.
"""

EXAMPLES = [
    {
        "about_me": "I am a 28 year old marketing professional.\nI have some savings and I'm interested in crypto investments.\nIs Bitcoin a good investment option?",
        "context": "El Salvador adopts Bitcoin as legal tender.\nRecent fluctuations in Bitcoin's price.\nRenewed interest from institutional investors."
    },
    {
        "about_me": "I am a 35 year old small business owner.\nI have a moderate risk tolerance and want to diversify my investments.\nShould I consider investing in renewable energy stocks?",
        "context": "Government announces increased focus on renewable energy.\nRenewable energy sector experiences consistent growth.\nFluctuations in oil prices impact energy market dynamics."
    },
    {
        "about_me": "I am a 45 year old IT professional nearing retirement.\nI am looking for stable investment options for my retirement funds.\nAre dividend-paying stocks a good choice for me?",
        "context": "Several blue-chip companies announce dividend increases.\nHistorical performance of dividend-paying stocks during economic downturns.\nCurrent low interest rate environment affects traditional savings options."
    },
    {
        "about_me": "I am a 29 year old freelance writer with variable income.\nI want to start investing but I'm unsure where to begin.\nAre robo-advisors a suitable option for me?",
        "context": "Robo-advisors gain popularity for their automated investment strategies.\nLow fees and accessibility make robo-advisors appealing for beginners.\nImportance of aligning investment choices with risk tolerance and financial goals."
    },
    {
        "about_me": "I am a 22 year old engineering student passionate about technology.\nI have a limited budget and want to invest in growth-oriented stocks.\nIs it a good idea to invest in emerging tech companies?",
        "context": "Recent IPOs of promising tech startups attract investor attention.\nPotential for rapid growth in emerging tech sectors like AI and clean energy.\nHigher volatility associated with investing in early-stage companies."
    },
    {
        "about_me": "I am a 40 year old healthcare professional planning for my children's education.\nI am risk-averse and prefer stable investment options.\nAre bonds or fixed-income securities suitable for my investment goals?",
        "context": "Fluctuations in interest rates impact bond prices and yields.\nGovernment bonds provide relative stability during market uncertainty.\nImportance of assessing investment time horizon and diversification."
    },
    {
        "about_me": "I am a 31 year old teacher looking to supplement my income through investments.\nI'm interested in real estate but have limited knowledge in this area.\nShould I consider real estate investment trusts (REITs)?",
        "context": "REITs offer exposure to real estate assets without direct ownership.\nIncome potential through dividends and potential for capital appreciation.\nImportance of understanding different types of REITs and their underlying assets."
    },
    {
        "about_me": "I am a 50 year old entrepreneur planning for retirement.\nI have experience in the tech industry and want to invest in individual stocks.\nShould I focus on well-established tech giants or emerging startups?",
        "context": "Established tech companies provide stability and dividend income.\nEmerging startups offer potential for higher growth but come with higher risk.\nImportance of balancing portfolio by diversifying across different tech sectors."
    },
    {
        "about_me": "I am a 26 year old graphic designer with a passion for sustainable living.\nI want to align my investments with my values.\nAre impact investing and socially responsible funds suitable options for me?",
        "context": "Growing interest in ESG (Environmental, Social, and Governance) investing.\nImpact funds focus on companies contributing positively to social and environmental issues.\nImportance of researching fund holdings and assessing their alignment with personal values."
    },
    {
        "about_me": "I am a 34 year old finance professional with a strong analytical background.\nI'm interested in diversifying my portfolio with alternative investments.\nShould I explore cryptocurrency investments to achieve this diversification?",
        "context": "Cryptocurrencies gain recognition as an emerging asset class.\nVolatility and regulatory uncertainties are inherent in the cryptocurrency market.\nImportance of understanding blockchain technology and researching specific cryptocurrencies."
    },
    {
        "about_me": "I am a 25 year old software engineer with a stable income.\nI want to start investing in stocks for long-term growth.\nWhere should I begin?",
        "context": "Tech stocks have shown strong growth in recent years.\nMarket volatility due to global economic conditions.\nImportance of diversifying across different sectors for risk management."
    },
    {
        "about_me": "I am a 32 year old business owner looking to invest my excess funds.\nReal estate has caught my interest, but I'm unsure about the market.",
        "context": "Real estate market experiences fluctuations based on supply and demand.\nLow mortgage rates stimulate demand, but economic downturns can impact property values.\nImportance of conducting thorough market research and understanding property cycles."
    },
    {
        "about_me": "I am a 40 year old healthcare professional.\nI'm concerned about inflation and its impact on my savings.\nWhat investment strategies can help hedge against inflation?",
        "context": "Inflation erodes purchasing power over time.\nHistorically, commodities like gold have been considered inflation hedges.\nInvesting in dividend-paying stocks and real estate can also provide protection against rising prices."
    },
    {
        "about_me": "I am a 29 year old social media manager with a passion for sustainability.\nAre there investment options that align with my values?",
        "context": "Growing interest in ESG (Environmental, Social, and Governance) investing.\nImpact funds focus on companies addressing environmental and social issues.\nImportance of researching companies' sustainability practices and evaluating their long-term impact."
    },
    {
        "about_me": "I am a 35 year old parent, and I want to secure my child's future education.\nHow can I invest effectively to cover education expenses?",
        "context": "529 savings plans offer tax advantages for education-related expenses.\nImportance of starting early to benefit from compounding growth.\nConsideration of the investment options within the 529 plan and their risk profiles."
    },
    {
        "about_me": "I am a 28 year old marketing professional.\nI'm interested in diversifying my investment portfolio.\nWhat are your thoughts on investing in Bitcoin?",
        "context": "Bitcoin experiences 15% price drop in the last week.\nElon Musk tweets about the environmental concerns of Bitcoin mining.\nThe Federal Reserve announces interest rate hike."
    },
    {
        "about_me": "I am a 45 year old small business owner.\nI have some extra funds and I'm considering investing in tech stocks.\nWhat do you think about Amazon's future prospects?",
        "context": "Amazon reports record-breaking Q2 revenue.\nRegulators announce increased scrutiny on big tech companies.\nJeff Bezos steps down as Amazon's CEO."
    },
    {
        "about_me": "I am a 35 year old software engineer.\nI'm looking to invest a lump sum amount for short-term gains.\nWhat's your opinion on the recent performance of Tesla?",
        "context": "Tesla announces plans for a new Gigafactory in Europe.\nElectric vehicle sales surge in the Asia-Pacific region.\nElon Musk sells 5% of his personal Tesla holdings."
    },
    {
        "about_me": "I am a 50 year old retiree.\nI want to ensure a stable income through my investments.\nDo you think dividend stocks are a good option at this time?",
        "context": "Several blue-chip companies announce dividend cuts.\nMarket analysts predict increased market volatility.\nThe Federal Reserve hints at continuing low interest rates."
    },
    {
        "about_me": "I am a 29 year old medical student.\nI want to start investing but I'm concerned about market risks.\nWhat's your take on stablecoins in the cryptocurrency market?",
        "context": "Regulators investigate Tether's reserve backing.\nStablecoin transactions surpass $1 trillion in the last month.\nMajor central banks explore the idea of digital currencies."
    },
    {
        "about_me": "I am a 32 year old marketing manager.\nI've been following the electric vehicle industry closely.\nDo you think it's a good idea to invest in a diverse EV portfolio?",
        "context": "New government incentives announced for electric vehicle adoption.\nBattery technology breakthrough reported by a leading research institute.\nRising competition in the EV market leads to price wars."
    },
    {
        "about_me": "I am a 40 year old real estate agent.\nI'm looking to invest some savings in the financial markets.\nWhat's your opinion on gold as a safe-haven investment?",
        "context": "Global geopolitical tensions escalate, leading to increased demand for gold.\nInflation rates show a gradual upward trend.\nCentral banks of several countries continue to stockpile gold reserves."
    },
    {
        "about_me": "I am a 25 year old graduate student.\nI want to explore investing in renewable energy companies.\nDo you think solar energy stocks are a good bet right now?",
        "context": "Global push for renewable energy leads to increased investment in solar projects.\nSolar panel manufacturing costs decrease due to technological advancements.\nGovernments announce new incentives for solar energy adoption."
    },
    {
        "about_me": "I am a 31 year old IT professional.\nI'm interested in long-term investments with sustainable growth.\nWhat's your take on investing in index funds?",
        "context": "Index funds continue to outperform actively managed funds over the last year.\nMarket volatility prompts investors to seek stable long-term options.\nFinancial experts emphasize the benefits of diversified portfolios."
    },
    {
        "about_me": "I am a 22 year old recent college graduate.\nI want to start investing, but I'm concerned about economic uncertainty.\nWhat do you think about investing in blue-chip stocks?",
        "context": "Blue-chip stocks show resilience during recent market downturns.\nCorporate earnings reports exceed expectations for several well-established companies.\nMarket analysts predict a potential market correction in the coming months."
    },
    {
        "about_me": "I am a 55 year old business owner.\nI'm considering investing a portion of my profits in the commodities market.\nWhat's your opinion on the current oil market situation?",
        "context": "Global oil demand rebounds as travel restrictions ease.\nOPEC+ countries agree to increase oil production in response to rising prices.\nEnvironmental concerns lead to increased exploration of alternative energy sources."
    },
    {
        "about_me": "I am a 27 year old freelance writer.\nI have some savings and I'm considering investing in individual stocks.\nWhat's your opinion on the future of electric car companies?",
        "context": "Electric vehicle companies announce plans for expanded charging infrastructure.\nNew government regulations favor electric vehicle adoption.\nSupply chain disruptions impact electric car production, affecting stock prices."
    },
    {
        "about_me": "I am a 33 year old teacher.\nI'm interested in ethical investments that align with my values.\nWhat are your thoughts on socially responsible mutual funds?",
        "context": "Socially responsible mutual funds experience steady growth in the past year.\nCompanies with strong environmental, social, and governance (ESG) practices outperform peers.\nInvestors show increased interest in sustainability-driven investments."
    },
    {
        "about_me": "I am a 48 year old pharmacist.\nI'm concerned about healthcare industry fluctuations.\nDo you think investing in pharmaceutical stocks is a wise decision?",
        "context": "Pharmaceutical companies announce breakthroughs in COVID-19 treatments.\nRegulatory agencies scrutinize drug pricing policies.\nMarket volatility increases due to uncertainties related to healthcare reforms."
    },
    {
        "about_me": "I am a 30 year old graphic designer.\nI want to invest in something with potential for high returns.\nWhat's your opinion on investing in startup companies?",
        "context": "Startup funding reaches record highs in the technology sector.\nVenture capitalists emphasize the importance of due diligence in startup investments.\nMany startups face challenges in scaling due to supply chain disruptions."
    },
    {
        "about_me": "I am a 36 year old nurse.\nI'm looking for low-risk investments to supplement my income.\nWhat do you think about investing in government bonds?",
        "context": "Government bond yields remain relatively stable in recent months.\nInflation concerns lead to discussions about potential interest rate hikes.\nCentral banks employ bond-buying programs to stimulate economic recovery."
    },

]


openai.api_key = os.environ["OPENAI_API_KEY"]

def build_prompt(example: Dict) -> str:

    return PROMPT_TEMPLATE.format(
        ABOUT_ME=example["about_me"],
        CONTEXT=example['context'],
    )

def run():

    output = []
    for example in tqdm(EXAMPLES):
        
        prompt = build_prompt(example)
        logger.info(f'{prompt=}')

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=100,
        )

        response = response['choices'][0]['text']
        logger.info(f'{response=}')

        output.append({
            **example,
            'response': response
        })

    # save output as json file
    import json
    with open(DATA_DIR / 'training_data.json', 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == '__main__':
    run()


