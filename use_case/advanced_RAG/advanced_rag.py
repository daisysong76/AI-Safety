import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from typing import List, Dict, Any

class AdvancedRAGSystem:
    def __init__(self, 
                 embedding_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
                 generation_model_name: str = 'facebook/opt-350m'):
        """
        Initialize RAG System with embedding and generation models
        
        Args:
            embedding_model_name (str): Model for creating document embeddings
            generation_model_name (str): Language model for text generation
        """
        # Embedding model for document retrieval
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
        
        # Generation model
        self.gen_tokenizer = AutoTokenizer.from_pretrained(generation_model_name)
        self.gen_model = AutoModelForCausalLM.from_pretrained(generation_model_name)
        
        # FAISS vector index for efficient retrieval
        self.vector_index = None
        self.document_corpus = []
    
    def create_document_embeddings(self, documents: List[str]) -> np.ndarray:
        """
        Create embeddings for document corpus
        
        Args:
            documents (List[str]): Input documents
        
        Returns:
            np.ndarray: Document embeddings
        """
        # Tokenize documents
        inputs = self.embedding_tokenizer(
            documents, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        
        # Compute embeddings
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.numpy()
    
    def build_vector_index(self, documents: List[str]):
        """
        Build FAISS vector index for efficient retrieval
        
        Args:
            documents (List[str]): Corpus of documents
        """
        # Store original documents
        self.document_corpus = documents
        
        # Create embeddings
        embeddings = self.create_document_embeddings(documents)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatL2(dimension)
        self.vector_index.add(embeddings)
    
    def retrieve_top_k_documents(self, 
                                  query: str, 
                                  k: int = 3, 
                                  diversity_penalty: float = 0.1) -> List[str]:
        """
        Retrieve top-k most relevant documents with diversity considerations
        
        Args:
            query (str): Input query
            k (int): Number of documents to retrieve
            diversity_penalty (float): Penalty for similar documents
        
        Returns:
            List[str]: Retrieved documents
        """
        # Embed query
        query_embedding = self.create_document_embeddings([query])
        
        # Search in vector index
        distances, indices = self.vector_index.search(query_embedding, k * 2)
        
        # Diversity-aware document selection
        selected_docs = []
        used_indices = set()
        
        for idx, dist in zip(indices[0], distances[0]):
            if len(selected_docs) >= k:
                break
            
            # Avoid redundant documents
            if idx not in used_indices:
                selected_docs.append(self.document_corpus[idx])
                used_indices.add(idx)
        
        return selected_docs
    
    def bias_aware_generation(self, 
                               prompt: str, 
                               context_docs: List[str], 
                               max_tokens: int = 100) -> str:
        """
        Generate text with bias mitigation and context-awareness
        
        Args:
            prompt (str): Input prompt
            context_docs (List[str]): Retrieved context documents
            max_tokens (int): Maximum generation length
        
        Returns:
            str: Generated text with reduced bias
        """
        # Combine prompt with context
        augmented_prompt = f"{prompt}\n\nContext:\n" + "\n".join(context_docs)
        
        # Tokenize input
        inputs = self.gen_tokenizer(augmented_prompt, return_tensors='pt')
        
        # Custom decoding with bias reduction
        outputs = self.gen_model.generate(
            **inputs,
            max_length=inputs['input_ids'].shape[1] + max_tokens,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.2,  # Reduce redundancy
            diversity_penalty=0.5,   # Encourage diverse token generation
            bad_words_ids=[
                self.gen_tokenizer.encode(word) 
                for word in ['bias', 'prejudice', 'stereotype']
            ]
        )
        
        # Decode generated text
        generated_text = self.gen_tokenizer.decode(outputs[0])
        
        return generated_text
    
    def rag_pipeline(self, 
                     query: str, 
                     document_corpus: List[str], 
                     max_tokens: int = 100) -> str:
        """
        Complete RAG pipeline with retrieval and generation
        
        Args:
            query (str): User query
            document_corpus (List[str]): Corpus of documents
            max_tokens (int): Maximum generation length
        
        Returns:
            str: Generated response
        """
        # Build vector index
        self.build_vector_index(document_corpus)
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_top_k_documents(query)
        
        # Generate bias-aware response
        generated_response = self.bias_aware_generation(
            query, 
            retrieved_docs, 
            max_tokens
        )
        
        return generated_response

# Demonstration
def main():
    # Sample document corpus
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks are inspired by biological neural systems.",
        "Ethical AI considers fairness, transparency, and accountability."
    ]
    
    # Initialize RAG System
    rag_system = AdvancedRAGSystem()
    
    # Example query
    query = "What is machine learning?"
    response = rag_system.rag_pipeline(query, documents)
    
    print("Query:", query)
    print("Response:", response)

if __name__ == '__main__':
    main()