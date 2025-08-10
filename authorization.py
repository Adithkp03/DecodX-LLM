from fastapi import Header, HTTPException, Depends
from config import SECRET_TOKEN

async def verify_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    token = authorization.split()[1]
    if token != SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")
