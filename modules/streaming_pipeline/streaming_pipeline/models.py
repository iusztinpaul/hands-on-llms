from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel


class News(BaseModel):
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


class Document(BaseModel):
    id: str
    group_key: Optional[str] = None
    metadata: Optional[dict] = {}
    text: Optional[list]
    chunks: Optional[list]
    embeddings: Optional[list] = []
