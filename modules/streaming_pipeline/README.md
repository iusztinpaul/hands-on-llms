# Streaming Pipeline

Real-time feature pipeline, that ingests data from Alpaca, computes the embeddings from the documents, and stores them into a serverless Vector DB.

## Table of Contents

- [1. Motivation](#1-motivation)
- [2. Install](#2-install)
    - [2.1 Dependencies](#21-dependencies)
- [3. Usage](#3-usage)


---

## 1. Motivation

...

## 2. Install

### 2.1. Dependencies

Main dependencies you have to install yourself:
* Python 3.10
* Poetry 1.5.1
* GNU Make 4.3

Installing all the other dependencies is as easy as running:
```shell
make install-debian # if you run a debian based OS (e.g., Ubuntu)
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

### 2.2 Bytewax - Waxctl

Installing Waxctl is very simple. You just need to download the binary corresponding to your operating system and architecture [here](https://bytewax.io/downloads/).

**Ubuntu:**
```shell
tar xvzf waxctl_0.9.2_linux_amd64.tar.gz
mkdir ~/.local/bin
mv waxctl ~/.local/bin

echo "export PATH=~/.local/bin:$PATH" > ~/.bashrc
source ~/.bashrc
``````



## 3. Usage

Run production streaming pipeline:
```shell
make run
```

Run dev streaming pipeline:
```shell
make run_dev
```

Run docker:
```shell
make build
source .env && make run_docker
```

Run a query in your vector DB:
```shell
make search PARAMS='--query_string "Should I invest in Tesla?"'
```

### 3.1 Deploy AWS
Configure your AWS CLI and run:
```shell
make deploy_aws
```
**NOTE:** [Here](https://stackoverflow.com/questions/15904095/how-to-check-whether-my-user-data-passing-to-ec2-instance-is-working) is how you can check the **output of the instance**.

To undeploy
```shell
make undeploy_aws
```

...
