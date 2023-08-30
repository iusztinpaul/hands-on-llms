from transformers import AutoModel, AutoTokenizer

from streaming_pipeline.models import Document

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to("cuda")


# create embedding and store in vector db
def compute_embeddings(document: Document):
    for chunk in document.chunks:
        inputs = tokenizer(
            chunk, padding=True, truncation=True, return_tensors="pt", max_length=384
        ).to("cuda")
        result = model(**inputs)
        embeddings = result.last_hidden_state[:, 0, :].cpu().detach().numpy()
        lst = embeddings.flatten().tolist()

        document.embeddings.append(lst)

    return document
