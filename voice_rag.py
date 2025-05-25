import numpy as np
import requests
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import threading
import queue
import time
import tempfile
import wave

# Voice imports
from faster_whisper import WhisperModel
import pyttsx3
import sounddevice as sd
import scipy.io.wavfile as wavfile

class VoiceRAG:
    def __init__(self, data_file="data.txt", model_name="all-MiniLM-L6-v2", top_k=3):
        self.data_file = data_file
        self.top_k = top_k
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Voice settings
        self.sample_rate = 16000  # FastWhisper optimal rate
        self.recording = False
        self.audio_queue = queue.Queue()
        
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(model_name)
        
        print("Loading document chunks...")
        self.chunks = self.load_chunks()
        
        print("Creating embeddings...")
        self.embeddings = self.create_embeddings()
        
        print("Loading FastWhisper model...")
        self.whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        
        print("Initializing TTS...")
        self.tts_engine = pyttsx3.init()
        # Set faster speech rate for low latency
        self.tts_engine.setProperty('rate', 200)  # Default is ~200, increase for faster
        self.tts_engine.setProperty('volume', 0.9)
        
        print(f"Voice RAG system ready with {len(self.chunks)} chunks!")
        print("🎤 Voice commands: 'record' to start listening, 'stop' to stop, 'quit' to exit")
    
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
    
    def record_audio(self, duration=5):
        """Record audio with low latency"""
        print(f"🎤 Recording for {duration} seconds... Speak now!")
        
        try:
            # Record audio
            audio_data = sd.rec(int(duration * self.sample_rate), 
                              samplerate=self.sample_rate, 
                              channels=1, 
                              dtype=np.float32)
            sd.wait()  # Wait for recording to complete
            
            # Convert to int16 for better compatibility
            audio_data = (audio_data * 32767).astype(np.int16)
            
            return audio_data.flatten()
            
        except Exception as e:
            print(f"Error recording audio: {e}")
            return None
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio using FastWhisper"""
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                wavfile.write(temp_file.name, self.sample_rate, audio_data)
                temp_path = temp_file.name
            
            # Transcribe with FastWhisper
            segments, info = self.whisper_model.transcribe(temp_path, language="en")
            
            # Extract text
            transcription = ""
            for segment in segments:
                transcription += segment.text + " "
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return transcription.strip()
            
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None
    
    def speak_text(self, text):
        """Convert text to speech with low latency"""
        try:
            # Run TTS in separate thread for non-blocking operation
            def tts_thread():
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            
            thread = threading.Thread(target=tts_thread)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            print(f"Error with TTS: {e}")
    
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
    
    def answer_question(self, question, voice_mode=False):
        """Main RAG pipeline: search + generate"""
        if not voice_mode:
            print(f"\nQuestion: {question}")
            print("-" * 50)
        
        # Find relevant chunks
        relevant_chunks = self.search_relevant_chunks(question)
        
        if not relevant_chunks:
            response = "No relevant information found in the document."
            if voice_mode:
                self.speak_text(response)
            return response
        
        # Display context (only in text mode for brevity)
        if not voice_mode:
            print("Context pulled from documents:")
            for i, chunk_data in enumerate(relevant_chunks, 1):
                chunk = chunk_data['chunk']
                similarity = chunk_data['similarity']
                print(f"\nChunk {i} (similarity: {similarity:.3f}):")
                print(f"{chunk[:200]}...")
        
        # Create context for LLM
        context_text = ""
        for i, chunk_data in enumerate(relevant_chunks, 1):
            chunk = chunk_data['chunk']
            context_text += f"Context {i}: {chunk}\n\n"
        
        # Create prompt for LLM
        prompt = f"""Based on the following context from the document, please answer the question concisely and clearly.

Context:
{context_text}

Question: {question}

Answer based only on the provided context. Keep the answer concise for voice output."""

        if not voice_mode:
            print("\nGenerating response...")
            print("-" * 50)
        
        # Get response from Ollama
        response = self.query_ollama(prompt)
        
        # Speak response in voice mode
        if voice_mode:
            print(f"🔊 Speaking: {response[:100]}...")
            self.speak_text(response)
        
        return response
    
    def voice_interaction(self):
        """Handle voice-based interaction"""
        print("\n🎤 Voice mode activated!")
        print("Say your question clearly. Recording will start automatically.")
        
        # Record audio
        audio_data = self.record_audio(duration=5)
        
        if audio_data is None:
            print("❌ Failed to record audio")
            return
        
        print("🎯 Transcribing...")
        
        # Transcribe
        question = self.transcribe_audio(audio_data)
        
        if not question:
            print("❌ Could not understand speech")
            self.speak_text("Sorry, I could not understand what you said.")
            return
        
        print(f"🎤 You asked: {question}")
        
        # Answer question with voice output
        self.answer_question(question, voice_mode=True)

def main():
    # Check if data file exists
    if not os.path.exists("data.txt"):
        print("No data.txt found. Please run 'python pdf.py' first to process a PDF.")
        return
    
    # Initialize Voice RAG system
    rag = VoiceRAG(top_k=3)
    
    if len(rag.chunks) == 0:
        print("No document chunks loaded. Please check your data file.")
        return
    
    print("\n" + "="*60)
    print("🎤 VOICE-ENABLED LOCAL RAG SYSTEM READY")
    print("="*60)
    print("Commands:")
    print("• Type 'voice' or 'v' - Start voice interaction")
    print("• Type your question - Text-based query")
    print("• Type 'quit' or 'exit' - Stop the system")
    print("Make sure Ollama is running: 'ollama serve'")
    print("-"*60)
    
    # Interactive loop
    while True:
        try:
            user_input = input("\n💬 Enter command or question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() in ['voice', 'v', 'record']:
                rag.voice_interaction()
                continue
            
            if not user_input:
                continue
            
            # Text-based question
            answer = rag.answer_question(user_input, voice_mode=False)
            print(f"\n🤖 Mistral Response:\n{answer}")
            print("\n" + "="*60)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()