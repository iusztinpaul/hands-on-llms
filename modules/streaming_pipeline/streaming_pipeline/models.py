import hashlib
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel
from unstructured.cleaners.core import (
    clean,
    clean_non_ascii_chars,
    replace_unicode_quotes,
)
from unstructured.partition.html import partition_html
from unstructured.staging.huggingface import chunk_by_attention_window

from streaming_pipeline.embeddings import EmbeddingModelSingleton


class NewsArticle(BaseModel):
    """
    Attributes:
        id (str): News article ID
        headline (str): Headline or title of the article
        summary (str): Summary text for the article (may be first sentence of content)
        author (str): Original author of news article
        created_at (datetime): Date article was created (RFC 3339)
        updated_at (datetime): Date article was updated (RFC 3339)
        url (Optional[str]): URL of article (if applicable)
        content (str): Content of the news article (might contain HTML)
        symbols (str): List of related or mentioned symbols
        source (str): Source where the news originated from (e.g. Benzinga)
    """

    id: float
    headline: str
    summary: str
    author: str
    created_at: datetime
    updated_at: datetime
    url: Optional[str]
    content: str
    symbols: List[str]
    source: str

    def to_document(self) -> "Document":
        document_id = hashlib.md5(self.content.encode()).hexdigest()
        document = Document(id=document_id)

        article_elements = partition_html(text=self.content)
        cleaned_content = clean_non_ascii_chars(
            replace_unicode_quotes(clean(" ".join([str(x) for x in article_elements])))
        )
        cleaned_headline = clean_non_ascii_chars(
            replace_unicode_quotes(clean(self.headline))
        )
        cleaned_summary = clean_non_ascii_chars(
            replace_unicode_quotes(clean(self.summary))
        )

        document.text = [cleaned_headline, cleaned_summary, cleaned_content]
        document.metadata["headline"] = cleaned_headline
        document.metadata["summary"] = cleaned_summary
        document.metadata["url"] = self.url
        document.metadata["symbols"] = self.symbols
        document.metadata["author"] = self.author
        document.metadata["created_at"] = self.created_at

        return document


class Document(BaseModel):
    id: str
    group_key: Optional[str] = None
    metadata: dict = {}
    text: list = []
    chunks: list = []
    embeddings: list = []

    def to_payloads(self) -> List[dict]:
        payloads = []
        for c in self.chunks:
            payload = self.metadata
            payload.update({"text": c})
            payloads.append(payload)

        return payloads

    def compute_chunks(self, model: EmbeddingModelSingleton) -> "Document":
        for item in self.text:
            chunked_item = chunk_by_attention_window(
                item, model.tokenizer, max_input_size=model.max_input_length
            )

            self.chunks.extend(chunked_item)

        return self

    def compute_embeddings(self, model: EmbeddingModelSingleton) -> "Document":
        for chunk in self.chunks:
            embedding = model(chunk, to_list=True)

            self.embeddings.append(embedding)

        return self
