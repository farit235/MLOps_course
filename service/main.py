from fastapi import FastAPI, Query
from pydantic import BaseModel

from pipeline import TaskExtractor


class TextQuery(BaseModel):
    text: str


def get_model_path():
    return "distilbert-base-uncased"


task_extractor = TaskExtractor(get_model_path())
app = FastAPI()


@app.post("/generate_tasks")
async def generate_tasks(text_query: TextQuery):
    return {"tasks": task_extractor.get_several_tasks(text_query.text)}
