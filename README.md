# HackRx LLM Project

## Overview
This project is a modular LLM-powered service to answer questions based on documents and integrates flight number agent.

## Structure
- `app.py`: FastAPI app entrypoint
- `config.py`: Configuration and secrets management
- `models.py`: Request and response schema models
- `authorization.py`: API token validation
- `flight_api.py`: Flight number API integration
- `text_extraction.py`: Document text extraction & chunking
- `retrieval.py`: BM25 and FAISS retrieval logic
- `embedding.py`: Embedding model handling
- `answer_generation.py`: LLM answer generation
- `utils.py`: Helper functions
- `tests/`: Unit and integration tests

## Setup
Install dependencies from `requirements.txt` and set environment variables for your API keys and secrets.

## Usage
Run the app with:
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

