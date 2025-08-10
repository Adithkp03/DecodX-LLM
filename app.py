import logging
import asyncio
from fastapi import FastAPI, HTTPException, Depends

from models import Req, Resp
from authorization import verify_token
from puzzle import fetch_flight_number_async
from text_extraction import extract_text, smarter_chunking
from retrieval import build_bm25_index, dual_pass_retrieve
from answer_generation import generate_answer
from embedding import embedding_model, executor
from config import DEFAULT_CONCURRENCY_LIMIT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hackrx")

app = FastAPI(title="HackRx LLM API", version="v2")

# Add CORS middleware as needed

async def process(doc_url: str, questions: list[str]) -> list[str]:
    answers = []
    flight_questions = [q for q in questions if "flight number" in q.lower()]
    other_questions = [q for q in questions if "flight number" not in q.lower()]

    for q in flight_questions:
        try:
            answer = await fetch_flight_number_async()
            answers.append(answer)
        except Exception as e:
            logger.error(f"Flight agent error for question '{q}': {e}")
            answers.append(f"Error retrieving flight number: {e}")

    if other_questions:
        try:
            text = await extract_text(doc_url)
            chunks = smarter_chunking(text, embedding_model=embedding_model)

            if not chunks:
                answers.extend(["Information not found in the policy."] * len(other_questions))
                return answers

            bm25 = build_bm25_index(chunks)
            sem = asyncio.Semaphore(DEFAULT_CONCURRENCY_LIMIT)

            async def answer_question(q: str) -> str:
                async with sem:
                    ctx_chunks = await dual_pass_retrieve(q, chunks, faiss_index=None, bm25=bm25)
                    if not ctx_chunks:
                        return "Information not found in the policy."
                    return await generate_answer(q, ctx_chunks)

            other_answers = await asyncio.gather(*[answer_question(q) for q in other_questions])
            answers.extend(other_answers)
        except Exception as e:
            logger.error(f"Document QA error: {e}")
            answers.extend([f"Error processing question: {e}"] * len(other_questions))

    return answers

@app.post("/hackrx/run", response_model=Resp)
async def run(req: Req, token=Depends(verify_token)):
    try:
        answers = await process(req.documents, req.questions)
        return {"answers": answers}
    except Exception as e:
        logger.exception(f"Internal error: {e}")
        raise HTTPException(500, f"Internal server error: {e}")

@app.get("/health")
async def health():
    return {"status": "ok", "multilingual": True, "dynamic_context": True}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)
