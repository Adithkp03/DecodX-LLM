import asyncio
import re
import logging
import google.generativeai as genai

from config import GEMINI_API_KEY, MAX_OUTPUT_TOKENS

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

executor = None  # inject or import from embedding.py / main app

logger = logging.getLogger("hackrx")

async def generate_answer(question: str, context_chunks: list) -> str:
    if not context_chunks:
        return "Information not found in the policy."

    context = "\n\n".join(context_chunks)
    prompt = f"""Answer the following question from the context provided.

Context:

{context}

Question: {question}

Answer:"""

    def _gen():
        try:
            res = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    top_p=0.8,
                    top_k=25
                ),
            )
            ans = re.sub(r'\s+', ' ', res.text.strip())
            return ans if ans else "Information not found in the policy."
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "Information not found in the policy."

    # Use event loop with executor injected or global
    return await asyncio.get_event_loop().run_in_executor(executor, _gen)
