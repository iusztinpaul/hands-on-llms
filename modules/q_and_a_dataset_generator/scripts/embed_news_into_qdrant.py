from typing import Dict, Optional, List

import hashlib
from pydantic import BaseModel
from unstructured.partition.html import partition_html
from unstructured.cleaners.core import clean, replace_unicode_quotes, clean_non_ascii_chars
from unstructured.staging.huggingface import chunk_by_attention_window
from unstructured.staging.huggingface import stage_for_transformers
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from src.paths import DATA_DIR
from src.logger import get_console_logger

NEWS_FILE = DATA_DIR / 'news_2023-01-01_2023-01-05.json'
QDRANT_COLLECTION_NAME = 'alpaca_news'
QDRANT_VECTOR_SIZE = 384

logger = get_console_logger()

# tokenizer and LLM we use to embed the document text
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# init qdrant client and collection where we store the news
from src.vector_db_api import get_qdrant_client, init_collection
qdrant_client = get_qdrant_client()
qdrant_client = init_collection(
    qdrant_client=qdrant_client,
    collection_name=QDRANT_COLLECTION_NAME,
    vector_size=QDRANT_VECTOR_SIZE,
)

class Document(BaseModel):
    id: str
    group_key: Optional[str] = None
    metadata: Optional[dict] = {}
    text: Optional[list] = []
    chunks: Optional[list] = []
    embeddings: Optional[list] = []

def parse_document(_data: Dict) -> Document:
    """"""
    document_id = hashlib.md5(_data['content'].encode()).hexdigest()
    document = Document(id = document_id)
    article_elements = partition_html(text=_data['content'])
    _data['content'] = clean_non_ascii_chars(replace_unicode_quotes(clean(" ".join([str(x) for x in article_elements]))))
    _data['headline'] = clean_non_ascii_chars(replace_unicode_quotes(clean(_data['headline'])))
    _data['summary'] = clean_non_ascii_chars(replace_unicode_quotes(clean(_data['summary'])))

    document.text = [_data['headline'], _data['summary'], _data['content']]
    document.metadata['headline'] = _data['headline']
    document.metadata['summary'] = _data['summary']
    # document.metadata['url'] = _data['url']
    # document.metadata['symbols'] = _data['symbols']
    # document.metadata['author'] = _data['author']
    document.metadata['date'] = _data['date']
    return document

def chunk(document: Document) -> Document:
    chunks = []
    for text in document.text:
        chunks += chunk_by_attention_window(
            text, tokenizer, max_input_size=QDRANT_VECTOR_SIZE)
    
    document.chunks = chunks
    return document

# create embedding and store in vector db
def embedding(document: Document) -> Document:
    for chunk in document.text:
        inputs = tokenizer(chunk,
                           padding=True,
                           truncation=True,
                           return_tensors="pt",
                           max_length=QDRANT_VECTOR_SIZE)
        
        result = model(**inputs)
        embeddings = result.last_hidden_state[:, 0, :].cpu().detach().numpy()
        lst = embeddings.flatten().tolist()
        document.embeddings.append(lst)
    return document


def build_payloads(doc: Document) -> List:
    payloads = []
    for chunk in doc.chunks:
        payload = doc.metadata
        payload.update({"text":chunk})
        payloads.append(payload)
    return payloads


def push_document_to_qdrant(doc: Document) -> None:

    from qdrant_client.models import PointStruct

    _payloads = build_payloads(doc)

    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=idx,
                vector=vector,
                payload=_payload
            )
            for idx, (vector, _payload) in enumerate(zip(doc.embeddings, _payloads))
        ]
    )

def process_one_document(_data: Dict) -> None:
    """"""
    doc = parse_document(_data)
    doc = chunk(doc)
    doc = embedding(doc)
    push_document_to_qdrant(doc)

    return doc

def embed_news_into_qdrant(news_data: List[Dict], n_processes: int = 1) -> None:
    """"""
    results = []
    if n_processes == 1:
        # sequential
        for _data in tqdm(news_data):
            result = process_one_document(_data)
            results.append(result)
    
    else:
        # parallel
        import multiprocessing

        # Create a multiprocessing Pool
        with multiprocessing.Pool(processes=n_processes) as pool:
            # Use tqdm to create a progress bar
            results = list(tqdm(pool.imap(process_one_document, news_data),
                                total=len(news_data),
                                desc="Processing",
                                unit="news"))

    breakpoint()

if __name__ == '__main__':
    """"""
    import json
    with open(NEWS_FILE, 'r') as json_file:
        news_data = json.load(json_file)

    embed_news_into_qdrant(
        news_data,
        n_processes=1,
    )



