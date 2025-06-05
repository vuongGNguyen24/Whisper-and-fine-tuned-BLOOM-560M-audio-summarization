from fastapi import FastAPI
from api.endpoints import transcribe, summarize

app = FastAPI()

app.include_router(transcribe.router)
app.include_router(summarize.router)

@app.get("/")
def root():
    return {"message": "API is running"}
