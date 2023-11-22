import os
from typing import Dict

import openai
from src.logger import get_console_logger
from src.paths import DATA_DIR
from tqdm import tqdm

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
        "context": "El Salvador adopts Bitcoin as legal tender.\nRecent fluctuations in Bitcoin's price.\nRenewed interest from institutional investors.",
    },
    {
        "about_me": "I am a 35 year old small business owner.\nI have a moderate risk tolerance and want to diversify my investments.\nShould I consider investing in renewable energy stocks?",
        "context": "Government announces increased focus on renewable energy.\nRenewable energy sector experiences consistent growth.\nFluctuations in oil prices impact energy market dynamics.",
    },
    {
        "about_me": "I am a 45 year old IT professional nearing retirement.\nI am looking for stable investment options for my retirement funds.\nAre dividend-paying stocks a good choice for me?",
        "context": "Several blue-chip companies announce dividend increases.\nHistorical performance of dividend-paying stocks during economic downturns.\nCurrent low interest rate environment affects traditional savings options.",
    },
    {
        "about_me": "I am a 29 year old freelance writer with variable income.\nI want to start investing but I'm unsure where to begin.\nAre robo-advisors a suitable option for me?",
        "context": "Robo-advisors gain popularity for their automated investment strategies.\nLow fees and accessibility make robo-advisors appealing for beginners.\nImportance of aligning investment choices with risk tolerance and financial goals.",
    },
    {
        "about_me": "I am a 22 year old engineering student passionate about technology.\nI have a limited budget and want to invest in growth-oriented stocks.\nIs it a good idea to invest in emerging tech companies?",
        "context": "Recent IPOs of promising tech startups attract investor attention.\nPotential for rapid growth in emerging tech sectors like AI and clean energy.\nHigher volatility associated with investing in early-stage companies.",
    },
    {
        "about_me": "I am a 40 year old healthcare professional planning for my children's education.\nI am risk-averse and prefer stable investment options.\nAre bonds or fixed-income securities suitable for my investment goals?",
        "context": "Fluctuations in interest rates impact bond prices and yields.\nGovernment bonds provide relative stability during market uncertainty.\nImportance of assessing investment time horizon and diversification.",
    },
    {
        "about_me": "I am a 31 year old teacher looking to supplement my income through investments.\nI'm interested in real estate but have limited knowledge in this area.\nShould I consider real estate investment trusts (REITs)?",
        "context": "REITs offer exposure to real estate assets without direct ownership.\nIncome potential through dividends and potential for capital appreciation.\nImportance of understanding different types of REITs and their underlying assets.",
    },
    {
        "about_me": "I am a 50 year old entrepreneur planning for retirement.\nI have experience in the tech industry and want to invest in individual stocks.\nShould I focus on well-established tech giants or emerging startups?",
        "context": "Established tech companies provide stability and dividend income.\nEmerging startups offer potential for higher growth but come with higher risk.\nImportance of balancing portfolio by diversifying across different tech sectors.",
    },
    {
        "about_me": "I am a 26 year old graphic designer with a passion for sustainable living.\nI want to align my investments with my values.\nAre impact investing and socially responsible funds suitable options for me?",
        "context": "Growing interest in ESG (Environmental, Social, and Governance) investing.\nImpact funds focus on companies contributing positively to social and environmental issues.\nImportance of researching fund holdings and assessing their alignment with personal values.",
    },
    {
        "about_me": "I am a 34 year old finance professional with a strong analytical background.\nI'm interested in diversifying my portfolio with alternative investments.\nShould I explore cryptocurrency investments to achieve this diversification?",
        "context": "Cryptocurrencies gain recognition as an emerging asset class.\nVolatility and regulatory uncertainties are inherent in the cryptocurrency market.\nImportance of understanding blockchain technology and researching specific cryptocurrencies.",
    },
    {
        "about_me": "I am a 25 year old software engineer with a stable income.\nI want to start investing in stocks for long-term growth.\nWhere should I begin?",
        "context": "Tech stocks have shown strong growth in recent years.\nMarket volatility due to global economic conditions.\nImportance of diversifying across different sectors for risk management.",
    },
    {
        "about_me": "I am a 32 year old business owner looking to invest my excess funds.\nReal estate has caught my interest, but I'm unsure about the market.",
        "context": "Real estate market experiences fluctuations based on supply and demand.\nLow mortgage rates stimulate demand, but economic downturns can impact property values.\nImportance of conducting thorough market research and understanding property cycles.",
    },
    {
        "about_me": "I am a 40 year old healthcare professional.\nI'm concerned about inflation and its impact on my savings.\nWhat investment strategies can help hedge against inflation?",
        "context": "Inflation erodes purchasing power over time.\nHistorically, commodities like gold have been considered inflation hedges.\nInvesting in dividend-paying stocks and real estate can also provide protection against rising prices.",
    },
    {
        "about_me": "I am a 29 year old social media manager with a passion for sustainability.\nAre there investment options that align with my values?",
        "context": "Growing interest in ESG (Environmental, Social, and Governance) investing.\nImpact funds focus on companies addressing environmental and social issues.\nImportance of researching companies' sustainability practices and evaluating their long-term impact.",
    },
    {
        "about_me": "I am a 35 year old parent, and I want to secure my child's future education.\nHow can I invest effectively to cover education expenses?",
        "context": "529 savings plans offer tax advantages for education-related expenses.\nImportance of starting early to benefit from compounding growth.\nConsideration of the investment options within the 529 plan and their risk profiles.",
    },
    {
        "about_me": "I am a 28 year old marketing professional.\nI'm interested in diversifying my investment portfolio.\nWhat are your thoughts on investing in Bitcoin?",
        "context": "Bitcoin experiences 15% price drop in the last week.\nElon Musk tweets about the environmental concerns of Bitcoin mining.\nThe Federal Reserve announces interest rate hike.",
    },
    {
        "about_me": "I am a 45 year old small business owner.\nI have some extra funds and I'm considering investing in tech stocks.\nWhat do you think about Amazon's future prospects?",
        "context": "Amazon reports record-breaking Q2 revenue.\nRegulators announce increased scrutiny on big tech companies.\nJeff Bezos steps down as Amazon's CEO.",
    },
    {
        "about_me": "I am a 35 year old software engineer.\nI'm looking to invest a lump sum amount for short-term gains.\nWhat's your opinion on the recent performance of Tesla?",
        "context": "Tesla announces plans for a new Gigafactory in Europe.\nElectric vehicle sales surge in the Asia-Pacific region.\nElon Musk sells 5% of his personal Tesla holdings.",
    },
    {
        "about_me": "I am a 50 year old retiree.\nI want to ensure a stable income through my investments.\nDo you think dividend stocks are a good option at this time?",
        "context": "Several blue-chip companies announce dividend cuts.\nMarket analysts predict increased market volatility.\nThe Federal Reserve hints at continuing low interest rates.",
    },
    {
        "about_me": "I am a 29 year old medical student.\nI want to start investing but I'm concerned about market risks.\nWhat's your take on stablecoins in the cryptocurrency market?",
        "context": "Regulators investigate Tether's reserve backing.\nStablecoin transactions surpass $1 trillion in the last month.\nMajor central banks explore the idea of digital currencies.",
    },
    {
        "about_me": "I am a 32 year old marketing manager.\nI've been following the electric vehicle industry closely.\nDo you think it's a good idea to invest in a diverse EV portfolio?",
        "context": "New government incentives announced for electric vehicle adoption.\nBattery technology breakthrough reported by a leading research institute.\nRising competition in the EV market leads to price wars.",
    },
    {
        "about_me": "I am a 40 year old real estate agent.\nI'm looking to invest some savings in the financial markets.\nWhat's your opinion on gold as a safe-haven investment?",
        "context": "Global geopolitical tensions escalate, leading to increased demand for gold.\nInflation rates show a gradual upward trend.\nCentral banks of several countries continue to stockpile gold reserves.",
    },
    {
        "about_me": "I am a 25 year old graduate student.\nI want to explore investing in renewable energy companies.\nDo you think solar energy stocks are a good bet right now?",
        "context": "Global push for renewable energy leads to increased investment in solar projects.\nSolar panel manufacturing costs decrease due to technological advancements.\nGovernments announce new incentives for solar energy adoption.",
    },
    {
        "about_me": "I am a 31 year old IT professional.\nI'm interested in long-term investments with sustainable growth.\nWhat's your take on investing in index funds?",
        "context": "Index funds continue to outperform actively managed funds over the last year.\nMarket volatility prompts investors to seek stable long-term options.\nFinancial experts emphasize the benefits of diversified portfolios.",
    },
    {
        "about_me": "I am a 22 year old recent college graduate.\nI want to start investing, but I'm concerned about economic uncertainty.\nWhat do you think about investing in blue-chip stocks?",
        "context": "Blue-chip stocks show resilience during recent market downturns.\nCorporate earnings reports exceed expectations for several well-established companies.\nMarket analysts predict a potential market correction in the coming months.",
    },
    {
        "about_me": "I am a 55 year old business owner.\nI'm considering investing a portion of my profits in the commodities market.\nWhat's your opinion on the current oil market situation?",
        "context": "Global oil demand rebounds as travel restrictions ease.\nOPEC+ countries agree to increase oil production in response to rising prices.\nEnvironmental concerns lead to increased exploration of alternative energy sources.",
    },
    {
        "about_me": "I am a 27 year old freelance writer.\nI have some savings and I'm considering investing in individual stocks.\nWhat's your opinion on the future of electric car companies?",
        "context": "Electric vehicle companies announce plans for expanded charging infrastructure.\nNew government regulations favor electric vehicle adoption.\nSupply chain disruptions impact electric car production, affecting stock prices.",
    },
    {
        "about_me": "I am a 33 year old teacher.\nI'm interested in ethical investments that align with my values.\nWhat are your thoughts on socially responsible mutual funds?",
        "context": "Socially responsible mutual funds experience steady growth in the past year.\nCompanies with strong environmental, social, and governance (ESG) practices outperform peers.\nInvestors show increased interest in sustainability-driven investments.",
    },
    {
        "about_me": "I am a 48 year old pharmacist.\nI'm concerned about healthcare industry fluctuations.\nDo you think investing in pharmaceutical stocks is a wise decision?",
        "context": "Pharmaceutical companies announce breakthroughs in COVID-19 treatments.\nRegulatory agencies scrutinize drug pricing policies.\nMarket volatility increases due to uncertainties related to healthcare reforms.",
    },
    {
        "about_me": "I am a 30 year old graphic designer.\nI want to invest in something with potential for high returns.\nWhat's your opinion on investing in startup companies?",
        "context": "Startup funding reaches record highs in the technology sector.\nVenture capitalists emphasize the importance of due diligence in startup investments.\nMany startups face challenges in scaling due to supply chain disruptions.",
    },
    {
        "about_me": "I am a 36 year old nurse.\nI'm looking for low-risk investments to supplement my income.\nWhat do you think about investing in government bonds?",
        "context": "Government bond yields remain relatively stable in recent months.\nInflation concerns lead to discussions about potential interest rate hikes.\nCentral banks employ bond-buying programs to stimulate economic recovery.",
    },
    {
        "about_me": "I'm a 42 year old accountant.\nI'm interested in diversifying my investment portfolio.\nWhat are your thoughts on investing in technology stocks?",
        "context": "Tech sector experiences a significant sell-off due to regulatory concerns.",
    },
    {
        "about_me": "I recently inherited a sum of money and want to invest it wisely.\nI'm open to moderate risk for potential gains.\nDo you think real estate is a good investment option?",
        "context": "The housing market shows signs of cooling down after a period of rapid growth.",
    },
    {
        "about_me": "I'm a 29 year old software developer with a high risk tolerance.\nI'm interested in exploring cryptocurrency investments.\nHow do you see the future of the crypto market?",
        "context": "Cryptocurrency market faces increased government scrutiny.",
    },
    {
        "about_me": "I'm nearing retirement and want to ensure my investments are secure.\nSteady income is my priority.\nWhat's your take on investing in dividend stocks?",
        "context": "Unemployment rates decrease, leading to consumer spending growth.",
    },
    {
        "about_me": "I'm a 25 year old entrepreneur interested in emerging technologies.\nI'm willing to take calculated risks for potential high returns.\nWhat's your opinion on investing in AI-related companies?",
        "context": "Biotech stocks surge following FDA approvals for innovative treatments.",
    },
    {
        "about_me": "I'm a 38 year old teacher looking to invest in socially responsible options.\nEnvironmental sustainability is important to me.\nDo you think renewable energy stocks align with my values?",
        "context": "Renewable energy companies report record-breaking profits.",
    },
    {
        "about_me": "I'm a 31 year old marketing executive seeking growth-oriented investments.\nI'm comfortable with moderate risk.\nHow do you view the potential of e-commerce companies?",
        "context": "Global trade tensions lead to market volatility.",
    },
    {
        "about_me": "I'm a 50 year old artist with irregular income, looking for stable investments.\nPreservation of capital is my main goal.\nWhat's your opinion on investing in government bonds?",
        "context": "The Federal Reserve announces plans to taper its bond-buying program.",
    },
    {
        "about_me": "I'm a 27 year old parent wanting to invest for my child's education.\nLong-term growth is important.\nHow do you see the future prospects of education technology companies?",
        "context": "The gig economy expands, driving demand for platform-based companies.",
    },
    {
        "about_me": "I'm a 45 year old lawyer looking to diversify my investment portfolio.\nI have a balanced risk appetite.\nWhat's your take on investing in a mix of index funds and bonds?",
        "context": "Precious metals prices surge amid geopolitical uncertainties.",
    },
    {
        "about_me": "I'm a 29 year old graphic designer.\nI want to start investing but I'm risk-averse.\nDo you think investing in blue-chip stocks is a safe option?",
        "context": "Several blue-chip companies announce dividend cuts.",
    },
    {
        "about_me": "I'm a 35 year old project manager.\nI'm looking for investment options that provide regular income.\nWhat's your opinion on investing in real estate investment trusts (REITs)?",
        "context": "The Federal Reserve hints at continuing low interest rates.",
    },
    {
        "about_me": "I'm a 42 year old sales representative.\nI'm considering investing in commodities.\nDo you think the recent surge in oil prices is sustainable?",
        "context": "OPEC+ countries agree to increase oil production in response to rising prices.",
    },
    {
        "about_me": "I'm a 28 year old writer.\nI want to explore investing in international markets.\nWhat's your take on investing in emerging market funds?",
        "context": "Global trade tensions lead to market volatility.",
    },
    {
        "about_me": "I'm a 50 year old financial analyst.\nI'm interested in growth investments with a long-term horizon.\nHow do you view the potential of electric vehicle companies?",
        "context": "Electric vehicle companies announce plans for expanded charging infrastructure.",
    },
    {
        "about_me": "I'm a 33 year old healthcare professional.\nI'm cautious about market volatility.\nWhat's your opinion on investing in stablecoin-backed funds?",
        "context": "Stablecoin transactions surpass $1 trillion in the last month.",
    },
    {
        "about_me": "I'm a 36 year old marketing consultant.\nI'm considering investing in technology startups.\nDo you think recent advancements in AI technology are driving startup growth?",
        "context": "Startup funding reaches record highs in the technology sector.",
    },
    {
        "about_me": "I'm a 45 year old educator.\nI'm interested in ethical investments.\nHow do you view the prospects of companies that prioritize social responsibility?",
        "context": "Companies with strong environmental, social, and governance (ESG) practices outperform peers.",
    },
    {
        "about_me": "I'm a 27 year old financial planner.\nI want to diversify my investment portfolio.\nWhat's your take on investing in global index funds?",
        "context": "Central banks employ bond-buying programs to stimulate economic recovery.",
    },
    {
        "about_me": "I'm a 31 year old scientist.\nI'm curious about the potential of biotech investments.\nDo you think advancements in gene therapy are impacting biotech company valuations?",
        "context": "Biotech stocks surge following FDA approvals for innovative treatments.",
    },
    {
        "about_me": "I'm a 40 year old pharmacist.\nI'm interested in long-term investments.\nDo you think dividend reinvestment plans (DRIPs) are a good strategy?",
        "context": "Corporate earnings reports exceed expectations for several well-established companies.",
    },
    {
        "about_me": "I'm a 26 year old data analyst.\nI'm looking for opportunities in the financial technology sector.\nWhat's your opinion on investing in payment processing companies?",
        "context": "New government regulations favor electric vehicle adoption.",
    },
    {
        "about_me": "I'm a 34 year old insurance agent.\nI'm interested in income-generating investments.\nWhat do you think about investing in high-yield bond funds?",
        "context": "Government bond yields remain relatively stable in recent months.",
    },
    {
        "about_me": "I'm a 48 year old engineer.\nI'm cautious about market fluctuations.\nHow do you view the stability of utility company stocks?",
        "context": "Renewable energy companies report record-breaking profits.",
    },
    {
        "about_me": "I'm a 29 year old designer.\nI'm considering investing in cryptocurrency.\nDo you think the recent market volatility in crypto is temporary?",
        "context": "Cryptocurrency market faces increased government scrutiny.",
    },
    {
        "about_me": "I'm a 43 year old entrepreneur.\nI'm open to moderate risk for potential high returns.\nHow do you view the future of artificial intelligence stocks?",
        "context": "Several tech companies announce breakthroughs in AI technology.",
    },
    {
        "about_me": "I'm a 37 year old nurse.\nI want to invest for my retirement.\nWhat's your opinion on target-date retirement funds?",
        "context": "The Federal Reserve announces plans to taper its bond-buying program.",
    },
    {
        "about_me": "I'm a 32 year old sales manager.\nI'm interested in growth-oriented investments.\nHow do you view the potential of renewable energy stocks?",
        "context": "Renewable energy companies report record-breaking profits.",
    },
    {
        "about_me": "I'm a 49 year old consultant.\nI want to explore socially responsible investing.\nWhat's your take on investing in clean energy funds?",
        "context": "Environmental concerns lead to increased exploration of alternative energy sources.",
    },
    {
        "about_me": "I'm a 30 year old lawyer.\nI'm looking to diversify my investment portfolio.\nHow do you view the prospects of investing in emerging market ETFs?",
        "context": "Global trade tensions lead to market volatility.",
    },
    {
        "about_me": "I'm a 36 year old software engineer.\nI want to invest in something with potential for high returns.\nWhat's your opinion on investing in emerging market stocks?",
        "context": "Global trade tensions lead to market volatility.",
    },
    {
        "about_me": "I'm a 45 year old marketing manager.\nI'm interested in exploring alternative investments.\nDo you think investing in peer-to-peer lending platforms is a good idea?",
        "context": "Central banks employ bond-buying programs to stimulate economic recovery.",
    },
    {
        "about_me": "I'm a 29 year old teacher.\nI want to start investing for the first time.\nWhat's your take on investing in index funds?",
        "context": "Index funds continue to outperform actively managed funds over the last year.",
    },
    {
        "about_me": "I'm a 33 year old sales representative.\nI'm considering investing in commodities.\nDo you think recent supply chain disruptions will impact commodity prices?",
        "context": "Supply chain disruptions impact electric car production, affecting stock prices.",
    },
    {
        "about_me": "I'm a 27 year old marketing executive.\nI'm interested in socially responsible investments.\nHow do you view the potential of sustainable agriculture companies?",
        "context": "Regulators announce increased scrutiny on sustainable investing claims.",
    },
    {
        "about_me": "I'm a 39 year old pharmacist.\nI'm cautious about market risks.\nWhat's your opinion on investing in gold as a hedge against inflation?",
        "context": "Inflation concerns lead to discussions about potential interest rate hikes.",
    },
    {
        "about_me": "I'm a 31 year old IT professional.\nI want to diversify my investment portfolio.\nHow do you view the prospects of investing in cryptocurrency ETFs?",
        "context": "Cryptocurrency market faces increased government scrutiny.",
    },
    {
        "about_me": "I'm a 47 year old financial consultant.\nI'm interested in long-term growth.\nDo you think investing in emerging market bonds is a viable option?",
        "context": "Global trade tensions lead to market volatility.",
    },
    {
        "about_me": "I'm a 28 year old artist.\nI'm looking to invest in something that aligns with my values.\nWhat's your take on investing in impact funds?",
        "context": "Companies with strong environmental, social, and governance (ESG) practices outperform peers.",
    },
    {
        "about_me": "I'm a 34 year old engineer.\nI'm interested in the renewable energy sector.\nHow do you view the potential of solar energy companies?",
        "context": "Renewable energy companies report record-breaking profits.",
    },
    {
        "about_me": "I'm a 30 year old writer.\nI want to start investing but I'm risk-averse.\nDo you think investing in blue-chip stocks is a safe option?",
        "context": "Several blue-chip companies announce dividend cuts.",
    },
    {
        "about_me": "I'm a 32 year old accountant.\nI want to invest in stable options with moderate growth potential.\nWhat's your opinion on investing in dividend-paying utility stocks?",
        "context": "Utility stocks show steady growth and offer reliable dividend payments.",
    },
    {
        "about_me": "I'm a 28 year old software developer.\nI'm interested in exploring the cryptocurrency market.\nHow do you view the potential of decentralized finance (DeFi) projects?",
        "context": "The DeFi sector experiences rapid growth and innovation.",
    },
    {
        "about_me": "I'm a 39 year old nurse.\nI'm concerned about healthcare industry fluctuations.\nWhat's your take on investing in pharmaceutical companies?",
        "context": "Pharmaceutical companies announce breakthroughs in new drug therapies.",
    },
    {
        "about_me": "I'm a 43 year old teacher.\nI want to invest for retirement and stability.\nDo you think investing in government bonds is a secure option?",
        "context": "Government bond yields remain relatively stable in recent months.",
    },
    {
        "about_me": "I'm a 25 year old marketing professional.\nI'm considering investing in emerging markets.\nHow do you view the growth potential of Asian tech companies?",
        "context": "Asian tech companies show rapid expansion and innovation.",
    },
    {
        "about_me": "I'm a 36 year old entrepreneur.\nI'm open to moderate risk for potential high returns.\nWhat's your opinion on investing in blockchain technology companies?",
        "context": "Blockchain technology gains traction in various industries.",
    },
    {
        "about_me": "I'm a 30 year old pharmacist.\nI'm interested in long-term investments.\nHow do you view the potential of dividend reinvestment plans (DRIPs)?",
        "context": "Corporate earnings reports exceed expectations for several well-established companies.",
    },
    {
        "about_me": "I'm a 29 year old writer.\nI want to explore investing in alternative energy.\nWhat's your take on investing in wind energy companies?",
        "context": "Renewable energy companies report record-breaking profits.",
    },
    {
        "about_me": "I'm a 47 year old IT professional.\nI'm looking for opportunities in the technology sector.\nHow do you view the future of cloud computing companies?",
        "context": "Several tech companies announce advancements in cloud computing services.",
    },
    {
        "about_me": "I'm a 38 year old financial consultant.\nI'm cautious about market volatility.\nDo you think investing in stablecoins is a safe option?",
        "context": "Stablecoin transactions surpass $1 trillion in the last month.",
    },
    {
        "about_me": "I'm a 28 year old software developer.\nI'm interested in exploring the cryptocurrency market.\nWhat's your opinion on investing in Bitcoin?",
        "context": "Bitcoin experiences 15% price drop in the last week.",
    },
    {
        "about_me": "I'm a 35 year old entrepreneur.\nI want to diversify my investments with some exposure to cryptocurrencies.\nHow do you view the potential of Ethereum?",
        "context": "Ethereum's upgrade to Ethereum 2.0 aims to improve scalability and sustainability.",
    },
    {
        "about_me": "I'm a 30 year old investor.\nI'm intrigued by the potential of new cryptocurrencies.\nWhat's your take on investing in altcoins?",
        "context": "Several new altcoins gain popularity due to innovative features and use cases.",
    },
    {
        "about_me": "I'm a 32 year old tech enthusiast.\nI'm considering investing in blockchain projects.\nHow do you view the growth potential of decentralized applications (dApps)?",
        "context": "Decentralized applications gain traction across various industries.",
    },
    {
        "about_me": "I'm a 29 year old finance professional.\nI'm cautious about market risks.\nWhat's your opinion on investing in stablecoins?",
        "context": "Stablecoin transactions surpass $1 trillion in the last month.",
    },
    {
        "about_me": "I'm a 27 year old cryptocurrency enthusiast.\nI'm interested in long-term investments.\nDo you think staking cryptocurrencies is a viable option?",
        "context": "Staking gains popularity as a way to earn passive income with cryptocurrencies.",
    },
    {
        "about_me": "I'm a 31 year old investor.\nI want to explore the potential of the NFT market.\nHow do you view the future of non-fungible tokens (NFTs)?",
        "context": "NFTs gain mainstream attention with high-profile sales and digital art collections.",
    },
    {
        "about_me": "I'm a 36 year old tech professional.\nI'm intrigued by the concept of privacy-focused cryptocurrencies.\nWhat's your take on investing in privacy coins?",
        "context": "Privacy coins face increased regulatory scrutiny due to concerns about illicit activities.",
    },
    {
        "about_me": "I'm a 33 year old investor.\nI'm considering investing in DeFi projects.\nHow do you view the potential of decentralized finance (DeFi) platforms?",
        "context": "The DeFi sector experiences rapid growth and innovation.",
    },
    {
        "about_me": "I'm a 25 year old cryptocurrency trader.\nI'm interested in short-term gains.\nWhat's your opinion on leveraging trading bots for cryptocurrency trading?",
        "context": "Cryptocurrency trading bots gain popularity as tools for automated trading strategies.",
    },
    {
        "about_me": "I'm a 30 year old investor.\nI want to explore opportunities in emerging markets.\nWhat's your opinion on investing in Asian stocks?",
        "context": "Asian tech companies show rapid expansion and innovation.",
    },
    {
        "about_me": "I'm a 28 year old entrepreneur.\nI'm interested in growth investments with a long-term horizon.\nHow do you view the potential of Latin American markets?",
        "context": "Latin American economies experience increased foreign investment and growth prospects.",
    },
    {
        "about_me": "I'm a 32 year old finance professional.\nI'm looking to diversify my investment portfolio.\nDo you think investing in African markets is a viable option?",
        "context": "African economies attract attention with improving business environments and resource abundance.",
    },
    {
        "about_me": "I'm a 35 year old investor.\nI want to explore opportunities in emerging markets.\nHow do you view the potential of Middle Eastern markets?",
        "context": "Middle Eastern countries invest heavily in diversifying economies and expanding infrastructure.",
    },
    {
        "about_me": "I'm a 29 year old tech enthusiast.\nI'm intrigued by the growth of emerging market tech sectors.\nWhat's your take on investing in Indian tech companies?",
        "context": "Indian tech companies show significant growth in software services and e-commerce.",
    },
    {
        "about_me": "I'm a 27 year old investor.\nI'm interested in growth investments with a long-term horizon.\nHow do you view the potential of Southeast Asian markets?",
        "context": "Southeast Asian economies benefit from rising consumer demand and regional integration.",
    },
    {
        "about_me": "I'm a 31 year old finance professional.\nI'm looking for opportunities in the real estate sector.\nWhat's your opinion on investing in property markets in Latin America?",
        "context": "Latin American real estate markets experience increased demand due to urbanization and tourism.",
    },
    {
        "about_me": "I'm a 26 year old investor.\nI want to explore opportunities in emerging markets.\nHow do you view the growth potential of Eastern European markets?",
        "context": "Eastern European economies attract investments with improving infrastructure and skilled workforce.",
    },
    {
        "about_me": "I'm a 34 year old entrepreneur.\nI'm interested in growth investments with a long-term horizon.\nDo you think investing in Middle Eastern tech startups is a good idea?",
        "context": "Middle Eastern tech startups receive significant venture capital investments.",
    },
    {
        "about_me": "I'm a 30 year old investor.\nI'm intrigued by the growth of renewable energy in emerging markets.\nWhat's your take on investing in solar energy projects in Africa?",
        "context": "African countries invest in solar energy projects to address energy demand and sustainability.",
    },
]


openai.api_key = os.environ["OPENAI_API_KEY"]


def build_prompt(example: Dict) -> str:
    return PROMPT_TEMPLATE.format(
        ABOUT_ME=example["about_me"],
        CONTEXT=example["context"],
    )


def run():
    output = []
    for example in tqdm(EXAMPLES):
        prompt = build_prompt(example)
        logger.info(f"{prompt=}")

        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=100,
        )

        response = response["choices"][0]["text"]
        logger.info(f"{response=}")

        output.append({**example, "response": response})

    # save output as json file
    import json

    with open(DATA_DIR / "training_data.json", "w") as f:
        json.dump(output, f, indent=4)


if __name__ == "__main__":
    run()
