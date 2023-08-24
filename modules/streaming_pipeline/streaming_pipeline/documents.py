from unstructured.partition.html import partition_html
from unstructured.cleaners.core import clean, replace_unicode_quotes, clean_non_ascii_chars
from unstructured.staging.huggingface import chunk_by_attention_window

import hashlib
from pydantic import BaseModel
from typing import Any, Optional

class Document(BaseModel):
    id: str
    group_key: Optional[str] = None
    metadata: Optional[dict] = {}
    text: Optional[list]
    chunks: Optional[list]
    embeddings: Optional[list] = []


# Clean the code and setup the dataclass
def parse_article(_data):
    document_id = hashlib.md5(_data['content'].encode()).hexdigest()
    document = Document(id = document_id)
    article_elements = partition_html(text=_data['content'])
    _data['content'] = clean_non_ascii_chars(replace_unicode_quotes(clean(" ".join([str(x) for x in article_elements]))))
    _data['headline'] = clean_non_ascii_chars(replace_unicode_quotes(clean(_data['headline'])))
    _data['summary'] = clean_non_ascii_chars(replace_unicode_quotes(clean(_data['summary'])))

    document.text = [_data['headline'], _data['summary'], _data['content']]
    document.metadata['headline'] = _data['headline']
    document.metadata['summary'] = _data['summary']
    document.metadata['url'] = _data['url']
    document.metadata['symbols'] = _data['symbols']
    document.metadata['author'] = _data['author']
    document.metadata['created_at'] = _data['created_at']
    return document

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# chunk the news article and summary
def chunk(document):
    chunks = []
    for text in document.text:
        chunks += chunk_by_attention_window(text, tokenizer, max_input_size=384)
    
    document.chunks = chunks
    return document
