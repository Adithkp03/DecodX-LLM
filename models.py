from pydantic import BaseModel
from typing import List

class Req(BaseModel):
    documents: str
    questions: List[str]

class Resp(BaseModel):
    answers: List[str]
