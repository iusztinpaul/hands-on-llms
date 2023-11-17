<div align="center">
    <h2>Hands-on LLMs Course </h2>
    <h1>Train and Deploy a Real-Time Financial Advisor</h1>
    <i>by <a href="https://github.com/iusztinpaul">Paul Iusztin</a> and <a href="https://github.com/Paulescu">Pau Labarta Bajo</a></i>
</div>

## Table of Contents

- [1. Building Blocks](#1-building-blocks)
- [2. Setup External Services](#2-setup-external-services)
- [3. Install & Usage](#3-install--usage)
- [4. Video lectures](#4-video-lectures)
- [5. License](#5-license)
- [6. Contributors & Teachers](#6-contributors--teachers)

------


## 1. Building Blocks

### Training pipeline

Training pipeline that supports fine-tuning an LLM on a proprietary Q&A dataset and storing it in a model registry. 

### Streaming real-time pipeline

Real-time feature pipeline that:
- ingests financial news from [Alpaca](https://alpaca.markets/docs/api-references/market-data-api/news-data/)
- transforms the news documents into embeddings in real-time using [Bytewax](https://github.com/bytewax/bytewax?utm_source=thepauls&utm_medium=partner&utm_content=github)
- stores the embeddings into the [Qdrant Vector DB](https://qdrant.tech/?utm_source=thepauls&utm_medium=partner&utm_content=github)

### Inference pipeline

Inference pipeline that uses [LangChain](https://github.com/langchain-ai/langchain) to create a chain that:
* downloads the fine-tuned model from [Comet's](https://www.comet.com?utm_source=thepauls&utm_medium=partner&utm_content=github) model registry
* takes user questions as input
* queries the [Qdrant Vector DB](https://qdrant.tech/?utm_source=thepauls&utm_medium=partner&utm_content=github) and enhances the prompt with related financial news
* calls the fine-tuned LLM for the final answer
* persists the chat history into memory 

<br/>

![architecture](media/architecture.png)


## 2. Setup External Services

Before diving into the modules, you have to set up a couple of additional tools for the course.

### 2.1. Alpaca
`financial news data source`

Follow this [document](https://alpaca.markets/docs/market-data/getting-started/), showing you how to create a FREE account, generate the API Keys, and put them somewhere safe.


### 2.2. Qdrant
`vector DB`

Go to [Qdrant](https://qdrant.tech/?utm_source=thepauls&utm_medium=partner&utm_content=github), create a FREE account, and follow [this document](https://qdrant.tech/documentation/cloud/authentication/?utm_source=thepauls&utm_medium=partner&utm_content=github) on how to generate the API Keys.


### 2.3. Comet ML
`ML platform`

Go to [Comet ML](https://www.comet.com/signup?utm_source=thepauls&utm_medium=partner&utm_content=github), create a FREE account, a project, and an API KEY. We will show you in every module how to add these credentials.


### 2.4. Beam
`cloud compute`

Go to [Beam](https://www.beam.cloud?utm_source=thepauls&utm_medium=partner&utm_content=github) and follow their quick setup/get started tutorial. You must create a FREE account, install their CLI and configure your credentials on your local machine.

- [Introduction guide](https://docs.beam.cloud/getting-started/introduction?utm_source=thepauls&utm_medium=partner&utm_content=github)
- [Installation guide](https://docs.beam.cloud/getting-started/installation?utm_source=thepauls&utm_medium=partner&utm_content=github)

#### Troubleshooting

When using Poetry, we had issues locating the Beam CLI when using it inside the Poetry virtual environment. To fix this, after installing Beam, create a symlink that points to Poetry's binaries, as follows:
 ```shell
  export COURSE_MODULE_PATH=<your-course-module-path> # e.g., modules/training_pipeline
  cd $COURSE_MODULE_PATH
  export POETRY_ENV_PATH=$(dirname $(dirname $(poetry run which python)))

  ln -s /usr/local/bin/beam ${POETRY_ENV_PATH}/bin/beam
 ```


 ### 2.5. AWS
 `cloud compute`

 Go to [AWS](https://aws.amazon.com/console/), create an account, and generate a pair of credentials.

 After, download and install their [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) and [configure it](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) with your credentials.


## 3. Install & Usage
Every module has its dependencies and scripts. In a production setup, every module would have its repository, but in this use case, for learning purposes, we put everything in one place:

Thus, check out the README for every module individually to see how to install & use it:
1. [q_and_a_dataset_generator](/modules/q_and_a_dataset_generator/)
2. [training_pipeline](/modules/training_pipeline/)
3. [streaming_pipeline](/modules/streaming_pipeline/)
4. [inference_pipeline](/modules/financial_bot/)

## 4. Video lectures

### 4.0 Intro to the course

<div align="center">
  <a href="https://www.youtube.com/watch?v=l4HTEf0_s70">
      <p>Click here to watch the video 🎬</p>
    <img src="media/youtube_thumbnails/00_intro.png" alt="Intro to the course" style="width:75%;">
  </a>
</div>


### 4.1 Fine-tuning our open-source LLM (overview)

<div align="center">
  <a href="https://www.youtube.com/watch?v=HcxwOYMmj40">
      <p>Click here to watch the video 🎬</p>
    <img src="media/youtube_thumbnails/01_fine_tuning_pipeline_overview.png" alt="Intro to the course" style="width:75%;">
  </a>
</div>

### 4.2 Fine-tuning our open-source LLM (Hands-on!)

<div align="center">
  <a href="https://www.youtube.com/watch?v=RS96R0dH0uE">
      <p>Click here to watch the video 🎬</p>
    <img src="media/youtube_thumbnails/02_fine_tuning_pipeline_hands_on.png" alt="Hands-on Fine Tuning an LLM" style="width:75%;">
  </a>
</div>

## 5. License

This course is an open-source project released under the MIT license. Thus, as long you distribute our LICENSE and acknowledge our work, you can safely clone or fork this project and use it as a source of inspiration for whatever you want (e.g., university projects, college degree projects, etc.).

## 6. Contributors & Teachers

<table>
  <tr>
    <td><img src="https://github.com/Paulescu.png" width="100" style="border-radius:50%;"/></td>
    <td>
      <strong>Pau Labarta Bajo | Senior ML & MLOps Engineer </strong><br />
      <i>Main teacher. The guy from the video lessons.</i><br /><br />
      <a href="https://www.linkedin.com/in/pau-labarta-bajo-4432074b/">LinkedIn</a><br />
      <a href="https://twitter.com/paulabartabajo_">Twitter/X</a><br />
      <a href="https://www.realworldml.xyz/subscribe">Real-World ML Newsletter</a><br />
      <a href="https://www.realworldml.xyz/subscribe">Real-World ML Site</a>
    </td>
  </tr>
  <tr>
    <td><img src="https://github.com/Joywalker.png" width="100" style="border-radius:50%;"/></td>
    <td>
      <strong>Alexandru Razvant | Senior ML Engineer </strong><br />
      <i>Second chef. The engineer behind the scenes.</i><br /><br />
      <a href="https://www.linkedin.com/in/arazvant/">LinkedIn</a><br />
      <a href="https://www.neuraleaps.com/">Neura Leaps</a>
    </td>
  </tr>
  <tr>
    <td><img src="https://github.com/iusztinpaul.png" width="100" style="border-radius:50%;"/></td>
    <td>
      <strong>Paul Iusztin | Senior ML & MLOps Engineer </strong><br />
      <i>Main chef. The guys who randomly pop in the video lessons.</i><br /><br />
      <a href="https://www.linkedin.com/in/pauliusztin/">LinkedIn</a><br />
      <a href="https://twitter.com/iusztinpaul">Twitter/X</a><br />
      <a href="https://pauliusztin.substack.com/">Decoding ML Newsletter</a><br />
      <a href="https://www.pauliusztin.me/">Personal Site | ML & MLOps Hub</a>
    </td>
  </tr>
</table>
