from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from predict import get_model_response


class Question(BaseModel):
    question: str


app = FastAPI()


@app.post("/")
async def create_question(question: Question):
    dict_question = question.dict()
    answer = get_model_response(dict_question)
    return answer
