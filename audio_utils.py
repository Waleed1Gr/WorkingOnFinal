import sounddevice as sd
import numpy as np
import threading
import queue
import time
from faster_whisper import WhisperModel

class RealTimeSTT:
    """Real-time Speech-to-Text with voice activity detection for ultra-low latency"""
    
    def __init__(self, model_size="base", device="cpu"):
        self.model = WhisperModel(model_size, device=device, compute_type="int8")
        self.sample_rate = 16000
        self.chunk_duration = 1.0  # Process audio every 1 second
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.background_thread = None
        
        # Voice Activity Detection parameters
        self.silence_threshold = 0.01  # Adjust based on environment
        self.min_speech_duration = 0.5  # Minimum speech length
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio status: {status}")
        
        # Add audio data to queue
        self.audio_queue.put(indata.copy())
    
    def detect_voice_activity(self, audio_chunk):
        """Simple voice activity detection"""
        energy = np.sqrt(np.mean(audio_chunk ** 2))
        return energy > self.silence_threshold
    
    def process_audio_stream(self):
        """Process audio stream in real-time"""
        audio_buffer = np.array([])
        speech_detected = False
        speech_start_time = 0
        
        while self.is_recording:
            try:
                # Get audio chunk from queue
                chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer = np.append(audio_buffer, chunk.flatten())
                
                # Check voice activity
                has_voice = self.detect_voice_activity(chunk)
                
                if has_voice and not speech_detected:
                    speech_detected = True
                    speech_start_time = time.time()
                    print("🎤 Speech detected...")
                
                elif not has_voice and speech_detected:
                    # End of speech detected
                    speech_duration = time.time() - speech_start_time
                    
                    if speech_duration > self.min_speech_duration and len(audio_buffer) > 0:
                        # Process the accumulated audio
                        yield self.transcribe_chunk(audio_buffer)
                    
                    # Reset
                    audio_buffer = np.array([])
                    speech_detected = False
                
                # Limit buffer size to prevent memory issues
                if len(audio_buffer) > self.sample_rate * 10:  # 10 seconds max
                    audio_buffer = audio_buffer[-self.sample_rate * 5:]  # Keep last 5 seconds
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing: {e}")
    
    def transcribe_chunk(self, audio_data):
        """Transcribe audio chunk"""
        try:
            # Normalize audio
            if len(audio_data) == 0:
                return ""
            
            audio_data = audio_data.astype(np.float32)
            
            # Transcribe
            segments, _ = self.model.transcribe(audio_data, language="en")
            
            text = ""
            for segment in segments:
                text += segment.text + " "
            
            return text.strip()
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def start_listening(self):
        """Start real-time listening"""
        self.is_recording = True
        
        # Start audio stream
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype=np.float32
        )
        
        self.stream.start()
        
        # Start processing thread
        self.background_thread = threading.Thread(target=self.process_audio_stream)
        self.background_thread.daemon = True
        self.background_thread.start()
        
        print("🎤 Real-time listening started. Speak naturally...")
        
        # Return generator for transcriptions
        return self.process_audio_stream()
    
    def stop_listening(self):
        """Stop listening"""
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        print("🛑 Stopped listening")

class FastTTS:
    """Optimized TTS for low latency"""
    
    def __init__(self):
        import pyttsx3
        self.engine = pyttsx3.init()
        
        # Optimize for speed
        self.engine.setProperty('rate', 250)  # Faster speech
        self.engine.setProperty('volume', 0.9)
        
        # Get available voices
        voices = self.engine.getProperty('voices')
        if voices:
            # Try to use a faster voice if available
            for voice in voices:
                if 'english' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
    
    def speak_async(self, text):
        """Non-blocking speech synthesis"""
        def speak():
            self.engine.say(text)
            self.engine.runAndWait()
        
        thread = threading.Thread(target=speak)
        thread.daemon = True
        thread.start()
        return thread
    
    def speak_interrupt(self, text):
        """Interrupt current speech and speak new text"""
        self.engine.stop()
        self.speak_async(text)

# Test functions
def test_realtime_stt():
    """Test real-time STT"""
    stt = RealTimeSTT(model_size="tiny")  # Use tiny model for speed
    
    try:
        transcription_generator = stt.start_listening()
        
        print("Speak now! Press Ctrl+C to stop...")
        
        for transcription in transcription_generator:
            if transcription:
                print(f"🎯 Transcribed: {transcription}")
        
    except KeyboardInterrupt:
        stt.stop_listening()
        print("Test completed!")

def test_fast_tts():
    """Test fast TTS"""
    tts = FastTTS()
    
    test_texts = [
        "Hello, this is a test of fast text to speech.",
        "The system is working correctly with low latency.",
        "Voice interaction is now available!"
    ]
    
    for text in test_texts:
        print(f"Speaking: {text}")
        thread = tts.speak_async(text)
        thread.join()  # Wait for completion
        time.sleep(0.5)

if __name__ == "__main__":
    print("Audio Utils Test")
    print("1. Test Real-time STT")
    print("2. Test Fast TTS")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        test_realtime_stt()
    elif choice == "2":
        test_fast_tts()
    else:
        print("Invalid choice")