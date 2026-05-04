from typing import Any, Dict, List

from pydantic import BaseModel


class InferInput(BaseModel):
    # Input Schema (Spec Lines 23-27)
    query: str
    metadata: Dict[str, Any]


class SourceSnippet(BaseModel):
    doc_id: str
    snippet: str


class InferOutput(BaseModel):
    # Output Schema (Spec Lines 28-36)
    response: str
    sources: List[SourceSnippet]
