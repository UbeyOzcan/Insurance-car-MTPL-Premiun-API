from typing import Annotated
from fastapi import FastAPI, Query
from pydantic.json_schema import SkipJsonSchema
app = FastAPI()


@app.get("/items/")
async def read_items(q: Annotated[list[str] | SkipJsonSchema[None], Query()] = None):
    query_items = {"q": q}
    return query_items