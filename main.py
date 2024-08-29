from fastapi import FastAPI
# from . import schemas
from pydantic import BaseModel


app=FastAPI()

class Blog(BaseModel):
    title:str
    body:str


@app.post('/blog')
def create(blog:Blog):
    return {'title':blog.title,'body':blog.body }

# @app.post('/items/',response_model=Blog)
# async def create_item(item:Blog):
#     return item