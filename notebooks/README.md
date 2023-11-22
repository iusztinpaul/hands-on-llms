## Exploratory Data Analysis of Finetuning Dataset

---
As in any ML problem, data used to train models is the indispensable component in the model's performance and value. Sample principle applies to Generative AI tasks too, more precisely in fine-tuning Large Language Models, one of the key components in this course.
#
Let's thus, go over a few bullet-points on why it might be important for you to consider analyzing your dataset before fine-tuning any LLM model.

Here's what EDA on the prompts of your dataset might provide details onto:

**Context Window Coverage**

ℹ️ Helps in selecting the optimal context window size for your dataset

ℹ️ Ensures that your model can capture the whole contextual meaning of your prompt.

ℹ️ Helps in accurately selecting a parameter size for your model (7B, 13B, 2B)

ℹ️ Helps in establishing the max_sequence_len, speeds-up fine-tuning due to lower

attention masking matmul’s.

********************Word Cloud********************

ℹ️ Visual representation of most representative words

ℹ️ Helps in preventing bias towards a specific theme (overfit)

ℹ️ Helps into processing/removing irrelevant frequent words

ℹ️ Could serve as a basic quality check of your overall dataset vocabulary

**************************************Tokens Distribution**************************************

ℹ️ Helps balancing lengths of your dataset prompts

ℹ️ Easy to check for the Gaussian Bell distribution of lengths

ℹ️ Helps identifying the samples that could be enhanced or summarised

ℹ️ Helps limiting the input prompt to specific size, allowing for completion sentence to be longer.


![EDA_KEYPOINTS](../media/eda_prompts_dataset.png)

### Get Started
Check the notebook at `notebooks/prompts_eda.ipynb` to inspect the charts and verbose details on what each chart provides.