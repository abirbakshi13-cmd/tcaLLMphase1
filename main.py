import os
from pathlib import Path

from fastapi import FastAPI
from openai import OpenAI
from dotenv import load_dotenv

from models import InferInput, InferOutput
from scripts.retrieval import build_vector_store, retrieve

app = FastAPI(title="TCA ML Workshop: Passenger Rights Advocate")

load_dotenv()

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
    snippets = retrieve(payload.query, VECTOR_STORE, top_k=3)
    if not snippets:
        return {"response": INSUFFICIENT_INFO, "sources": []}

    context = "\n\n".join(
        f"[DOC: {item.doc_id}]\n{item.snippet}" for item in snippets
    )
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    system_prompt = (
        "You are a Passenger Rights Advocate helping with airline travel "
        "legal issues based on their Contract of Carriage. Answer clearly "
        "and concisely, and do not cite sources. You must answer ONLY using "
        "the provided context. If the answer is not present in the context, "
        f"respond exactly with: {INSUFFICIENT_INFO}"
    )
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuery:\n{payload.query}"},
        ],
    )
    response_text = completion.choices[0].message.content or ""
    if response_text.strip() == INSUFFICIENT_INFO:
        return {"response": INSUFFICIENT_INFO, "sources": []}

    sources = [
        {"doc_id": item.doc_id, "snippet": item.snippet} for item in snippets
    ]
    return {"response": response_text, "sources": sources}
