# Financial Assistant Bot

Inference pipeline that uses [LangChain](https://github.com/langchain-ai/langchain) to create a chain that:
* downloads the fine-tuned model from [Comet's](https://www.comet.com?utm_source=thepauls&utm_medium=partner&utm_content=github) model registry
* takes user questions as input
* queries the [Qdrant Vector DB](https://qdrant.tech/?utm_source=thepauls&utm_medium=partner&utm_content=github) and enhances the prompt with related financial news
* calls the fine-tuned LLM for the final answer
* persists the chat history into memory 


## Table of Contents

- [1. Motivation](#1-motivation)
- [2. Install](#2-install)
    - [2.1. Dependencies](#21-dependencies)
    - [2.2. Qdrant](#21-qdrant)
    - [2.3. Beam](#21-beam)
- [3. Usage](#3-usage)
    - [3.1. Local](#31-local)
    - [3.2. Local](#32-deploy-to-beam)
    - [3.3. Linting & Formatting](#34-linting--formatting)

# 1. Motivation

The inference pipeline defines how the user interacts with all the components we've built so far. We will combine all the components and make the actual financial assistant chatbot.

Thus, using [LangChain](https://github.com/langchain-ai/langchain), we will create a series of chains that:
* download & load the fine-tuned model from [Comet's](https://www.comet.com?utm_source=thepauls&utm_medium=partner&utm_content=github) model registry
* take the user's input, embed it, and query the [Qdrant Vector DB](https://qdrant.tech/?utm_source=thepauls&utm_medium=partner&utm_content=github) to extract related financial news
* build the prompt based on the user input, financial news context, and chat history
* call the LLM
* persist the history in memory

Also, the final step is to put the financial assistant to good use and deploy it as a serverless RESTful API using [Beam](https://www.beam.cloud?utm_source=thepauls&utm_medium=partner&utm_content=github). 

# 2. Install 

# 2.1. Dependencies

Main dependencies you have to install yourself:
* Python 3.10
* Poetry 1.5.1
* GNU Make 4.3

Install dependencies:
```shell
make install
```

For developing run:
```shell
make install_dev
```

Prepare credentials:
```shell
cp .env.example .env
```
--> and complete the `.env` file with your credentials.

### 2.2. Qdrant

You must create a FREE account in Qdrant and generate the `QDRANT_API_KEY` and `QDRANT_URL` environment variables. After, be sure to add them to your `.env` file.

-> [Check out this document to see how.](https://qdrant.tech/documentation/cloud/authentication/?utm_source=thepauls&utm_medium=partner&utm_content=github)


### 2.3. Beam
`optional step in case you want to use Beam` 

Create and configure a free Beam account to deploy it as a serverless RESTful API and show it to your friends. You will pay only for what you use. 

-> [Create a Beam account & configure it.](https://www.beam.cloud?utm_source=thepauls&utm_medium=partner&utm_content=github)


# 3. Usage

# 3.1. Local

Run bot locally:
```shell
make run
```

Run bot locally in dev mode:
```shell
make run_dev
```

# 3.2. Deploy to Beam

Deploy the bot under a RESTful API using Beam:
```shell
make deploy_beam
```

Deploy the bot under a RESTful API using Beam in dev mode:
```shell
make deploy_beam_dev
```

Make a request to the bot calling the RESTful API:
```shell
export BEAM_DEPLOYMENT_ID=<BEAM_DEPLOYMENT_ID>
export BEAM_AUTH_TOKEN=<BEAM_AUTH_TOKEN>

make call_restful_api DEPLOYMENT_ID=${BEAM_DEPLOYMENT_ID} TOKEN=${BEAM_AUTH_TOKEN} 
```

# 3.3. Linting & Formatting

**Check** the code for **linting** issues:
```shell
make lint_check
```

**Fix** the code for **linting** issues (note that some issues can't automatically be fixed, so you might need to solve them manually):
```shell
make lint_fix
```

**Check** the code for **formatting** issues:
```shell
make format_check
```

**Fix** the code for **formatting** issues:
```shell
make format_fix
```
