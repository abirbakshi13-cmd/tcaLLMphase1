import os

from fastapi import FastAPI
from openai import OpenAI
from dotenv import load_dotenv

from models import InferInput, InferOutput

app = FastAPI(title="TCA ML Workshop: Passenger Rights Advocate")

load_dotenv()


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
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    system_prompt = (
        "You are a Passenger Rights Advocate helping with airline travel "
        "legal issues based on their Contract of Carriage. Answer clearly "
        "and concisely, and do not cite sources."
    )
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": payload.query},
        ],
    )
    response_text = completion.choices[0].message.content or ""
    return {"response": response_text, "sources": []}
