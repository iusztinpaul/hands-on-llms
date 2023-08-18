# Semi-automatic Q&A dataset generation

## Context

Fine-tuning is about adjusting the model weights to maximize performance on a narrowly defined specific task, for example, provide the best possible financial advice.

In a real-world project, we would hire a team of financial experts, to bootstrap an initial dataset of pairs (question, answer). In this tutorial, we will folow a semi-automatic approach, and use a general LLM, like ChatGPT, to bootstrap a reasonable training set.

This dataset should resemble as much as possible the actual questions, and answers we expect, from this model once deployed. This is the dataset we will use to **fine-tune** our LLM.

## Example

```
question = """
You are a financial advisor and your role is to provide me with the best financial advise, taking into account my financial goals, situation, and the most recent and relevant financial news.

** News from [DAYS AGO 1]
[NEWS 1]

** News from [DAYS AGO 2]
[NEWS 2]

** News from [DAYS AGO 3]
[NEWS 4]
"""

answer = """
bla bla bla bala
"""
```

> **A bit more about prompt engineering**
> Here is a recent prompt engineering idea we can use with ChatGP
> https://twitter.com/jeremyphoward/status/1689464587077509120

## Quick set up

* Set up virtual env using Poetry
    ```
    $ make init
    ```

* Run the init script for environment variables
    ```
    $ . ./set_env_variables.sh
    ```

* Get around `18k` news from January 2023 from Alpaca into a JSON file:
    ```
    $ make download
    ```

* Push this JSON file to Qdrant DB as embeddings
    ```
    $ make embed
    ```

* Generate a sample of training data
    [PENDING]


## TODOs
- [x] Dump historical news data from Alpaca into Qdrant Serverless Vector DB.
    More precisely, we will dump the news title, and the timestamp.
- [ ] Generate 10 prompt templates, for 10 different points in time (aka total of 100 prompts)
- [ ] Get 100 reasonable responses from ChatGPT.
- [ ] Push the dataset to HuggingFace datasets.