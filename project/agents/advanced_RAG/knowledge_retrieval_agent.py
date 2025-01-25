from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI

class KnowledgeRetrievalAgent:
    def __init__(self, faiss_index_path):
        vectorstore = FAISS.load_local(faiss_index_path)
        self.qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.5), retriever=vectorstore.as_retriever())

    def retrieve_knowledge(self, query):
        return self.qa_chain.run(query)
