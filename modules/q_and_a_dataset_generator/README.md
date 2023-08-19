# Semi-automatic Q&A dataset generation

## Context

Fine-tuning is about adjusting the model weights to maximize performance on a narrowly defined specific task, for example, provide the best possible financial advice.

In a real-world project, we would hire a team of financial experts, to bootstrap an initial dataset of pairs (question, answer). In this tutorial, we will folow a semi-automatic approach, and use a general LLM, like ChatGPT, to bootstrap a reasonable training set.

This dataset should resemble as much as possible the actual questions, and answers we expect, from this model once deployed. This is the dataset we will use to **fine-tune** our LLM.


## Quick set up

* Set up virtual env using Poetry
    ```
    $ make init
    ```

* Run the init script for environment variables
    ```
    $ . ./set_env_variables.sh
    ```

* Generate a sample of training data
    ```
    $ make training-data
    ```

## Not used here but might be useful later on

Unused pieces of code that can be useful later on, for example, to backfill the feature store
or the vector db.

* Get around `18k` news from January 2023 from Alpaca into a JSON file:
    ```
    $ make download
    ```

* Push this JSON file to Qdrant DB as embeddings
    ```
    $ make embed
    ```

## References
> **A bit more about prompt engineering**
> Here is a recent prompt engineering idea we can use with ChatGP
> https://twitter.com/jeremyphoward/status/1689464587077509120
