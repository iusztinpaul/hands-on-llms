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


## 3. Usage

...
