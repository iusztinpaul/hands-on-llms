import hashlib

from unstructured.cleaners.core import (
    clean,
    clean_non_ascii_chars,
    replace_unicode_quotes,
)
from unstructured.partition.html import partition_html
from unstructured.staging.huggingface import chunk_by_attention_window

from streaming_pipeline.embeddings import tokenizer
from streaming_pipeline.models import Document, NewsArticle


# Clean the code and setup the dataclass
def parse_article(article: NewsArticle):
    document_id = hashlib.md5(article.content.encode()).hexdigest()

    document = Document(id=document_id)
    article_elements = partition_html(text=article.content)
    article.content = clean_non_ascii_chars(
        replace_unicode_quotes(clean(" ".join([str(x) for x in article_elements])))
    )
    article.headline = clean_non_ascii_chars(
        replace_unicode_quotes(clean(article.headline))
    )
    article.summary = clean_non_ascii_chars(
        replace_unicode_quotes(clean(article.summary))
    )

    document.text = [article.headline, article.summary, article.content]
    document.metadata["headline"] = article.headline
    document.metadata["summary"] = article.summary
    document.metadata["url"] = article.url
    document.metadata["symbols"] = article.symbols
    document.metadata["author"] = article.author
    document.metadata["created_at"] = article.created_at

    return document


# chunk the news article and summary
def chunk(document: Document):
    chunks = []
    for text in document.text:
        chunks += chunk_by_attention_window(text, tokenizer, max_input_size=384)

    document.chunks = chunks
    
    return document
