from fastapi import FastAPI
from pydantic import BaseModel
app=FastAPI()
#Decorator path operation
@app.get('/blog')
def index(limit,published:bool=True):
    # return published
    #only get
    if published:
        return {'data':f'{limit} blogs are published from the db list'}
    else:
        return {'data':f'{limit} blogs are not published from the db list'}
@app.get('/blog/unpublished')
def about():    
    return {'data':'All unpublished blog'}

@app.get('/blog/{id}')
def about(id:int):    
    return {'data':id}

@app.get('/blog/{id}/comments')
def comments(id):    
    return {'data':{'comments',"mume dedunga choco"}}

class Blog(BaseModel):
    title:str
    body:str
    published:bool

@app.post('/blog2')
def create_blog(blog:Blog):
    # return request
    return {'data':f'Blog created with title {blog.title}'}
