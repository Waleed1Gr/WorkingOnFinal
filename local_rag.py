import numpy as np
import requests
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

class LocalRAG:
    def __init__(self, data_file="data.txt", model_name="all-MiniLM-L6-v2", top_k=3):
        self.data_file = data_file
        self.top_k = top_k
        self.ollama_url = "http://localhost:11434/api/generate"
        
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(model_name)
        
        print("Loading document chunks...")
        self.chunks = self.load_chunks()
        
        print("Creating embeddings...")
        self.embeddings = self.create_embeddings()
        
        print(f"RAG system ready with {len(self.chunks)} chunks!")
    
    def load_chunks(self):
        """Load text chunks from file"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                chunks = [line.strip() for line in f if line.strip()]
            return chunks
        except FileNotFoundError:
            print(f"Data file '{self.data_file}' not found. Please run pdf.py first.")
            return []
        except Exception as e:
            print(f"Error loading chunks: {e}")
            return []
    
    def create_embeddings(self):
        """Create embeddings for all chunks"""
        if not self.chunks:
            return np.array([])
        
        embeddings = self.embedding_model.encode(self.chunks)
        return embeddings
    
    def search_relevant_chunks(self, query):
        """Find most relevant chunks for the query"""
        if len(self.chunks) == 0:
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[::-1][:self.top_k]
        
        relevant_chunks = []
        for idx in top_indices:
            relevant_chunks.append({
                'chunk': self.chunks[idx],
                'similarity': similarities[idx]
            })
        
        return relevant_chunks
    
    def query_ollama(self, prompt, model="mistral"):
        """Send query to local Ollama instance"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(self.ollama_url, json=payload)
            
            if response.status_code == 200:
                return response.json()['response']
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Cannot connect to Ollama. Make sure Ollama is running (ollama serve)"
        except Exception as e:
            return f"Error querying Ollama: {e}"
    
    def answer_question(self, question):
        """Main RAG pipeline: search + generate"""
        print(f"\nQuestion: {question}")
        print("-" * 50)
        
        # Find relevant chunks
        relevant_chunks = self.search_relevant_chunks(question)
        
        if not relevant_chunks:
            return "No relevant information found in the document."
        
        # Display context
        print("Context pulled from documents:")
        context_text = ""
        for i, chunk_data in enumerate(relevant_chunks, 1):
            chunk = chunk_data['chunk']
            similarity = chunk_data['similarity']
            print(f"\nChunk {i} (similarity: {similarity:.3f}):")
            print(f"{chunk[:200]}...")
            context_text += f"Context {i}: {chunk}\n\n"
        
        # Create prompt for LLM
        prompt = f"""Based on the following context from the document, please answer the question.

Context:
{context_text}

Question: {question}

Answer based only on the provided context. If the context doesn't contain enough information to answer the question, say so."""

        print("\nGenerating response...")
        print("-" * 50)
        
        # Get response from Ollama
        response = self.query_ollama(prompt)
        return response

def main():
    # Check if data file exists
    if not os.path.exists("data.txt"):
        print("No data.txt found. Please run 'python pdf.py' first to process a PDF.")
        return
    
    # Initialize RAG system
    rag = LocalRAG(top_k=3)  # Change top_k here if needed
    
    if len(rag.chunks) == 0:
        print("No document chunks loaded. Please check your data file.")
        return
    
    print("\n" + "="*60)
    print("LOCAL RAG SYSTEM READY")
    print("="*60)
    print("Type 'quit' or 'exit' to stop")
    print("Make sure Ollama is running: 'ollama serve'")
    print("-"*60)
    
    # Interactive loop
    while True:
        try:
            question = input("\nEnter your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            # Get answer
            answer = rag.answer_question(question)
            print(f"\nMistral Response:\n{answer}")
            print("\n" + "="*60)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()