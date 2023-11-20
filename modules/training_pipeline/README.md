# Training / Fine-tuning Pipeline 

Training pipeline that:
- loads a proprietary Q&A dataset 
- fine-tunes an open-source LLM using QLoRA
- logs the training experiments on [Comet ML's](https://www.comet.com?utm_source=thepauls&utm_medium=partner&utm_content=github) experiment tracker & the inference results on [Comet ML's](https://www.comet.com?utm_source=thepauls&utm_medium=partner&utm_content=github) LLMOps dashboard
- stores the best model on [Comet ML's](https://www.comet.com/site/products/llmops/?utm_source=thepauls&utm_medium=partner&utm_content=github) model registry

The **training pipeline** is **deployed** using [Beam](https://docs.beam.cloud/getting-started/quickstart?utm_source=thepauls&utm_medium=partner&utm_content=github) as a serverless GPU infrastructure.

## Table of Contents

- [1. Motivation](#1-motivation)
- [2. Install](#2-install)
    - [2.1 Dependencies](#21-dependencies)
    - [2.2. Beam](#22-beam)
- [3. Usage](#3-usage)
    - [3.1. Train](#31-train)
    - [3.2. Inference](#32-inference)
    - [3.3. PEP8 Linting & Formatting](#33-pep8-linting--formatting)

-------

# 1. Motivation

The best way to specialize an LLM on your specific task is to fine-tune it on a small dataset coupled to your business use case.

In this case, we will use the finance dataset generated using the `q_and_a_dataset_generator` to specialize the LLM in responding to investing questions.

<br/>

![architecture](../../media/training_pipeline_architecture.png)


# 2. Install

## 2.1. Dependencies

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


## 2.2. Beam
`optional step in case you want to use Beam` 

-> [Create a Beam account, install its CLI, and configure it.](https://www.beam.cloud?utm_source=thepauls&utm_medium=partner&utm_content=github)

In addition to setting up Beam, you have to go to your [Beam account](https://www.beam.cloud?utm_source=thepauls&utm_medium=partner&utm_content=github) and create a volume, as follows:
- go to the `Volumes` section
- click create `New Volume` (in the top right corner)
- choose `Volume Name = qa_dataset` and `Volume Type = Shared`

After, run the following command to upload the Q&A dataset to the Beam volume you just created.
```shell
make upload_dataset_to_beam
```
Finally, check out that your [**qa_dataset** Beam volume](https://www.beam.cloud/dashboard/volumes/qa_dataset?utm_source=thepauls&utm_medium=partner&utm_content=github) contains the uploaded data. 

# 3. Usage

## 3.1. Train 

### Local

For debugging or to test that everything is working fine, run the following to train the model on a small subset of the dataset:
```shell
make dev_train_local
```

For training on the whole dataset, run the following:
```shell
make train_local
```

### Using Beam

Similar to the local training, for debugging or testing, run:
```shell
make dev_train_beam
```

For training on the whole dataset, run:
```shell
make train_beam
```

## 3.2. Inference

### Local

Testing or debugging:
```shell
make dev_infer_local
```

The whole deal:
```shell
make infer_local
```

### Using Beam

Testing or debugging:
```shell
make dev_infer_beam
```

The whole deal:
```shell
make infer_beam
```

## 3.3. PEP8 Linting & Formatting

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
