import pyaudio
import numpy as np
import whisper



def audio_catching():
            
            # Settings
            RATE = 16000
            CHUNK = 1024
            SILENCE_THRESHOLD = 500  # adjust if needed
            SILENCE_CHUNKS = 30      # stop after ~2 sec of silence (30 * 1024/16000 ≈ 2s)

            # Whisper model
            model = whisper.load_model("base")

            # Start audio stream
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16,
                            channels=1,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)

            print("🎙️ Speak now. Listening will stop after silence...")

            frames = []
            silent_chunks = 0

            while True:
                data = stream.read(CHUNK)
                frames.append(data)
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                volume = np.abs(audio_chunk).mean()

                if volume < SILENCE_THRESHOLD:
                    silent_chunks += 1
                else:
                    silent_chunks = 0

                if silent_chunks > SILENCE_CHUNKS:
                    break

            print("🛑 Silence detected. Transcribing...")

            # Stop recording
            stream.stop_stream()
            stream.close()
            p.terminate()

            # Prepare audio for Whisper
            audio_data = b''.join(frames)
            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe
            result = model.transcribe(audio_np)
            

            return  result["text"]


if __name__=="__main__":
       print(audio_catching())
             
