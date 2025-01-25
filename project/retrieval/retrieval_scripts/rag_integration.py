from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI

def setup_rag_pipeline():
    # Load FAISS index
    vectorstore = FAISS.load_local("faiss_index")
    llm = OpenAI(temperature=0.5)

    # Create RAG pipeline
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return qa_chain

if __name__ == "__main__":
    qa_chain = setup_rag_pipeline()
    question = "Why are women underrepresented in STEM fields?"
    print(qa_chain.run(question))
