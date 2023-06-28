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
