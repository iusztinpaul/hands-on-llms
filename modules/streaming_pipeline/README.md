# Streaming Pipeline

Real-time feature pipeline that:
- ingests financial news from [Alpaca](https://alpaca.markets/docs/api-references/market-data-api/news-data/)
- transforms the news documents into embeddings in real-time using [Bytewax](https://github.com/bytewax/bytewax?utm_source=thepauls&utm_medium=partner&utm_content=github)
- stores the embeddings into the [Qdrant Vector DB](https://qdrant.tech/?utm_source=thepauls&utm_medium=partner&utm_content=github)

## Table of Contents

- [1. Motivation](#1-motivation)
- [2. Install](#2-install)
    - [2.1 Dependencies](#21-dependencies)
    - [2.2. Alpaca](#22-alpaca)
    - [2.3. Qdrant](#23-qdrant)
    - [2.4. AWS CLI](#24-aws-cli)
- [3. Usage](#3-usage)
    - [3.1. Local](#31-local)
    - [3.2. Docker](#32-docker)
    - [3.3. Deploy to AWS](#33-deploy-to-aws)


---

## 1. Motivation

The best way to ingest real-time knowledge into an LLM without retraining the LLM too often is by using RAG.

To implement RAG at inference time, you need a vector DB always synced with the latest available data.

The role of this streaming pipeline is to listen 24/7 to available financial news from [Alpaca](https://alpaca.markets/docs/api-references/market-data-api/news-data/), process the news in real-time using [Bytewax](https://github.com/bytewax/bytewax?utm_source=thepauls&utm_medium=partner&utm_content=github), and store the news in the [Qdrant Vector DB](https://qdrant.tech/?utm_source=thepauls&utm_medium=partner&utm_content=github) to make the information available for RAG.

## 2. Install

### 2.1. Dependencies

Main dependencies you have to install yourself:
* Python 3.10
* Poetry 1.5.1
* GNU Make 4.3

Installing all the other dependencies is as easy as running:
```shell
make install
```

When developing run:
```shell
make install_dev
```

Prepare credentials:
```shell
cp .env.example .env
```
--> and complete the `.env` file with your credentials. We will show you below how to generate the credentials for **Alpaca** and **Qdrant** ↓ . 

### 2.2. Alpaca

All you have to do for Alpaca is create a FREE account and generate the `ALPACA_API_KEY` and `ALPACA_API_SECRET` API Keys. After, be sure to add them to your `.env` file. 

-> [Check out this document for step-by-step instructions.](https://alpaca.markets/docs/market-data/getting-started/)

### 2.3. Qdrant

Same as for Alpaca, you must create a FREE account in Qdrant and generate the `QDRANT_API_KEY` and `QDRANT_URL` environment variables. After, be sure to add them to your `.env` file.

-> [Check out this document to see how.](https://qdrant.tech/documentation/cloud/authentication/?utm_source=thepauls&utm_medium=partner&utm_content=github)

### 2.4. AWS CLI
`optional step in case you want to deploy the streaming pipeline to AWS`

First, install [AWS CLI 2.11.22](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

Secondly, configure the [credentials of your AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html). 


## 3. Usage

### 3.1. Local

Run production streaming pipeline in `real-time` mode:
```shell
make run_real_time
```

To populate the vector DB you can ingest historical data by running the streaming pipeline in `batch` mode:
```shell
make run_batch
```

Run the streaming pipeline in `real-time` and `development` modes:
```shell
make run_real_time_dev
```

Run the streaming pipeline in `batch` and `development` modes:
```shell
make run_batch_dev
```

Run a query in your vector DB:
```shell
make search PARAMS='--query_string "Should I invest in Tesla?"'
```
You can replace the `--query_string` with any question you want.

### 3.2. Docker

Build the Docker image:
```shell
make build
```

Run the streaming pipeline in `real-time` mode inside the Docker image:
```shell
source .env && make run_docker
```


### 3.3. Deploy to AWS
First, be sure that the `credentials` of your AWS CLI are configured.

After, run the following to deploy the streaming pipeline to an AWS EC2 machine: 
```shell
make deploy_aws
```

**NOTE:** You can log in to the AWS console, go to the EC2s section, and you can see your machine running.

To check the state of the deployment, run:
```shell
make info_aws
```

To remove the EC2 machine, run:
```shell
make undeploy_aws
```
