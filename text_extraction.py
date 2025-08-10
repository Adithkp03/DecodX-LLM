import re
import asyncio
import requests
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from typing import List
from config import OVERLAP_WORDS, DEFAULT_DESIRED_CHUNK_WORD_LEN, DEFAULT_MIN_CHUNK_WORD_LEN
from sklearn.cluster import AgglomerativeClustering

executor = ThreadPoolExecutor(max_workers=16)

def contains_url(text: str) -> bool:
    return bool(re.search(r'https?://\S+', text))

def is_pdf_url(url: str) -> bool:
    url_lower = url.lower()
    return url_lower.endswith(".pdf") or "pdf" in url_lower.split("?")[0].split(".")[-1]

session = requests.Session()

async def extract_text(url: str) -> str:
    if is_pdf_url(url):
        def extract_pdf_inner():
            r = session.get(url, timeout=30)
            r.raise_for_status()
            with fitz.open(stream=r.content, filetype="pdf") as doc:
                return " ".join(page.get_text() for page in doc if page.get_text().strip())
        text = await asyncio.get_event_loop().run_in_executor(executor, extract_pdf_inner)
    else:
        def extract_webpage_inner():
            r = session.get(url, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            return soup.get_text(separator=' ')
        text = await asyncio.get_event_loop().run_in_executor(executor, extract_webpage_inner)
    return re.sub(r'\s+', ' ', text).strip()

def split_into_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    return paragraphs

def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def add_overlap_between_chunks(chunks: List[str], overlap_word_count: int = OVERLAP_WORDS) -> List[str]:
    overlapped_chunks = []
    prev_words = []
    for chunk in chunks:
        current_words = chunk.split()
        if prev_words:
            combined_words = prev_words[-overlap_word_count:] + current_words
        else:
            combined_words = current_words
        overlapped_chunks.append(" ".join(combined_words))
        prev_words = current_words
    return overlapped_chunks

# You will need to import or inject your embedding model here when calling smarter_chunking.
def smarter_chunking(text: str,
                     desired_chunk_word_len: int = DEFAULT_DESIRED_CHUNK_WORD_LEN,
                     min_chunk_word_len: int = DEFAULT_MIN_CHUNK_WORD_LEN,
                     embedding_model=None) -> List[str]:
    paragraphs = split_into_paragraphs(text)
    if not paragraphs or all(len(p.split()) < min_chunk_word_len for p in paragraphs):
        paragraphs = split_into_sentences(text)

    embeddings = embedding_model.encode(paragraphs, convert_to_numpy=True)
    total_words = sum(len(p.split()) for p in paragraphs)
    n_clusters = max(1, total_words // desired_chunk_word_len)
    n_clusters = min(n_clusters, len(paragraphs))
    if n_clusters == 1:
        chunks = [" ".join(paragraphs)]
    else:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_ids = clusterer.fit_predict(embeddings)
        clusters = {}
        for i, cid in enumerate(cluster_ids):
            clusters.setdefault(cid, []).append(paragraphs[i])
        chunks = [" ".join(clusters[cid]).strip() for cid in sorted(clusters.keys())]

    chunks = add_overlap_between_chunks(chunks, overlap_word_count=OVERLAP_WORDS)
    return chunks
