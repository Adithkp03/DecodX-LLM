from fastapi import Header, HTTPException, Depends
from config import SECRET_TOKEN
import logging

logger = logging.getLogger(__name__)

async def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    token = authorization.split()[1]
    logger.info(f"verify_token received: {token}")
    logger.info(f"SECRET_TOKEN from env: {SECRET_TOKEN}")
    if token != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
