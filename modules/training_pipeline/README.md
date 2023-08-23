# Training / Fine-tuning Pipeline 

# Table of Contents

- [1. Motivation](#1-motivation)
- [2. Install](#2-install)
- [3. Usage](#3-usage)
    - [3.1. Train](#31-train)
    - [3.2. Inference](#32-inference)
    - [3.3. PEP8 Linting & Formatting](#33-pep8-linting--formatting) 

## 1. Motivation

The best way to specialize an LLM on your specific task is to fine-tune it on a small dataset coupled to your business use case.

In this case, we will use the finance dataset generated using the `q_and_a_dataset_generator` to specialize the LLM in responding to investing questions.


## 2. Install

Main dependencies you have to install yourself:
* Python 3.10
* Poetry 1.5.1
* GNU Make 4.3

Install all the dependencies is as easy as running:
```shell
make install
```

For developing run:
```shell
make install_dev
```

## 3. Usage

### 3.1. Train 

#### Local

For debugging or to test that everything is working fine, run the following to train the model on a small subset of the dataset:
```shell
make dev_train_local
```

For training on the whole dataset, run the following:
```shell
make train_local
```

#### Using Beam

Similar to the local training, for debugging or testing, run:
```shell
make dev_train_beam
```

For training on the whole dataset, run:
```shell
make train_beam
```

### 3.2. Inference

#### Local

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

### 3.3. PEP8 Linting & Formatting

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
