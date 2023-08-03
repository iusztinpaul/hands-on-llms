<div align="center">
    <h2>Hands-on LLMOps</h2>
    <h1>Train and Deploy a Real-Time Financial Advisor</h1>
    <i>by <a href="https://github.com/iusztinpaul">Paul Iusztin</a> and <a href="https://github.com/Paulescu">Pau Labarta Bajo</a></i>
    <!-- <i><a href="https://www.comet.com/signup?utm_source=pau&utm_medium=partner&utm_content=github">CometML</a></i> + <i><a href="https://www.cerebrium.ai?utm_source=pau&utm_medium=partner&utm_content=github">Cerebrium</a></i> = 🚀 -->
</div>

## Building blocks

### Training pipeline
- [ ] Fine-tune Falcon 7B using [the FinAQ dataset](https://paperswithcode.com/dataset/finqa).
    - It seems that 1 GPU is enough if we use [Lit-Parrot](https://lightning.ai/pages/blog/falcon-a-guide-to-finetune-and-inference/)

### Real-time data pipeline
- [ ] Build real-time feature pipeline, that ingests data form Alpaca, computes embeddings, and stores them into a serverless Vector DB.

### Inference pipeline
- [ ] REST API for inference, that
    1. receives a question (e.g. "Is it a good time to invest in renewable energy?"),
    2. finds the most relevant documents in the VectorDB (aka context)
    3. sends a prompt with question and context to our fine-tuned Falcon and return response.

## Usage

### Run Training on Beam
Go to Beam and create a new Volume called `train_finqa_dataset`, pick `Persistent` as the type, and finally choose the `train_finqa` app (#TODO: Automate this step.)

Upload dataset on a Beam volume:
```shell
beam volume upload train_finqa_dataset dataset -a train_finqa
```

Start the training on Beam:
```shell
cd modules
beam run tools/train_finqa.py:train
```

### Export Poetry Requirements to .txt
This will export all the poetry requirements, except the `dev` group, into a `requirements.txt` file:
```shell
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

### Run Notebooks Server
First expose the virtual environment as a notebook kernel:
```shell
python -m ipykernel install --user --name hands-on-llms --display-name "hands-on-llms"
```
Now run the notebook server:
```shell
jupyter notebook notebooks/ --ip 0.0.0.0 --port 8888
```

 ## Notes
 ### SSH Poetry
 When working through SSH, before using `poetry add ...` run the following command to unlock the keyring.
 ```shell
 export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
 ```

 ### Fix Beam using Poetry 
 ```shell
  ln -s /usr/local/bin/beam /home/pauliusztin/.cache/pypoetry/virtualenvs/training-6xkSxa8Q-py3.11/bin/beam
 ```
