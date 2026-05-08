import logging
import os
import time
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI
from openai import OpenAI
from dotenv import load_dotenv

from models import InferInput, InferOutput
from scripts.retrieval import build_vector_store, retrieve

app = FastAPI(title="TCA ML Workshop: Passenger Rights Advocate")

load_dotenv()

logger = logging.getLogger("monitoring")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

METRICS: Dict[str, float] = {
    "total_requests": 0,
    "total_latency_ms": 0,
    "source_relevance_hits": 0,
    "hallucination_count": 0,
}
FAILURE_CASES: List[Dict[str, str]] = []

DATA_DIR = Path(__file__).resolve().parent / "data"
VECTOR_STORE = build_vector_store(str(DATA_DIR))
INSUFFICIENT_INFO = "Insufficient information based on available documents."


@app.get("/")
def read_root():
    return {
        "message": (
            "Welcome to the TCA ML Workshop: "
            "Passenger Rights Advocate API"
        )
    }


@app.get("/health")
def health_check():
    # Health Endpoint (Spec Line 48)
    return {"status": "ok"}


@app.post("/infer", response_model=InferOutput)
def infer(payload: InferInput):
    # Input Schema (Spec Lines 23-27)
    # Output Schema (Spec Lines 28-36)
    # Inference Layer requirement: access LLM via API (Spec 6, 10)
    # Using Gemini via OpenAI-compatibility layer for Week 1 Inference Flow.
    # Monitoring Workflow requirement (Spec Line 97)
    start_time = time.perf_counter()
    snippets = retrieve(payload.query, VECTOR_STORE, top_k=3)
    context = "\n\n".join(
        f"[DOC: {item.doc_id}]\n{item.snippet}" for item in snippets
    )
    if not snippets:
        response_text = INSUFFICIENT_INFO
        latency_ms = (time.perf_counter() - start_time) * 1000
        _update_metrics(latency_ms=latency_ms, source_relevance=False, hallucinated=False)
        logger.info(
            (
                "INFER | Latency: %.2fms | Query: %s... | Status: %s"
                % (
                    latency_ms,
                    payload.query[:50],
                    "Fallback" if response_text == INSUFFICIENT_INFO else "Success",
                )
            ),
            extra={
                "user_query": payload.query,
                "retrieved_context": context,
                "llm_response": response_text,
                "latency_ms": round(latency_ms, 2),
            },
        )
        return {"response": INSUFFICIENT_INFO, "sources": []}
    client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url=os.getenv("GEMINI_BASE_URL"),
    )
    system_prompt = (
        "You are a Passenger Rights Advocate helping with airline travel "
        "legal issues based on their Contract of Carriage. Answer clearly "
        "and concisely, and do not cite sources. You must answer ONLY using "
        "the provided context. If the answer is not present in the context, "
        f"respond exactly with: {INSUFFICIENT_INFO}"
    )
    completion = client.chat.completions.create(
        model="gemini-3-flash-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuery:\n{payload.query}"},
        ],
    )
    response_text = completion.choices[0].message.content or ""
    latency_ms = (time.perf_counter() - start_time) * 1000
    if response_text.strip() == INSUFFICIENT_INFO:
        _update_metrics(latency_ms=latency_ms, source_relevance=False, hallucinated=False)
        logger.info(
            (
                "INFER | Latency: %.2fms | Query: %s... | Status: %s"
                % (
                    latency_ms,
                    payload.query[:50],
                    "Fallback" if response_text == INSUFFICIENT_INFO else "Success",
                )
            ),
            extra={
                "user_query": payload.query,
                "retrieved_context": context,
                "llm_response": INSUFFICIENT_INFO,
                "latency_ms": round(latency_ms, 2),
            },
        )
        return {"response": INSUFFICIENT_INFO, "sources": []}

    sources = [
        {"doc_id": item.doc_id, "snippet": item.snippet} for item in snippets
    ]
    hallucinated = _detect_hallucination(response_text, context)
    _update_metrics(
        latency_ms=latency_ms,
        source_relevance=True,
        hallucinated=hallucinated,
    )
    _record_failure_case(
        query=payload.query,
        context=context,
        response=response_text,
        hallucinated=hallucinated,
    )
    logger.info(
        (
            "INFER | Latency: %.2fms | Query: %s... | Status: %s"
            % (
                latency_ms,
                payload.query[:50],
                "Fallback" if response_text == INSUFFICIENT_INFO else "Success",
            )
        ),
        extra={
            "user_query": payload.query,
            "retrieved_context": context,
            "llm_response": response_text,
            "latency_ms": round(latency_ms, 2),
        },
    )
    return {"response": response_text, "sources": sources}


def _update_metrics(latency_ms: float, source_relevance: bool, hallucinated: bool) -> None:
    METRICS["total_requests"] += 1
    METRICS["total_latency_ms"] += latency_ms
    if source_relevance:
        METRICS["source_relevance_hits"] += 1
    if hallucinated:
        METRICS["hallucination_count"] += 1


def _detect_hallucination(response_text: str, context: str) -> bool:
    if not context.strip():
        return response_text.strip() != INSUFFICIENT_INFO

    response_terms = {term for term in response_text.lower().split() if len(term) > 4}
    context_terms = {term for term in context.lower().split() if len(term) > 4}
    if not response_terms:
        return False

    overlap_ratio = len(response_terms & context_terms) / len(response_terms)
    return overlap_ratio < 0.1


def _record_failure_case(
    query: str,
    context: str,
    response: str,
    hallucinated: bool,
) -> None:
    if not hallucinated:
        return

    FAILURE_CASES.append(
        {
            "type": "hallucination_or_mismatch",
            "query": query,
            "context": context,
            "response": response,
        }
    )
    if len(FAILURE_CASES) > 10:
        FAILURE_CASES.pop(0)
