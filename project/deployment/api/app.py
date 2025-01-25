from fastapi import FastAPI
from retrieval.rag_integration import setup_rag_pipeline

app = FastAPI()
qa_chain = setup_rag_pipeline()

@app.get("/ask")
def ask(question: str):
    response = qa_chain.run(question)
    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
