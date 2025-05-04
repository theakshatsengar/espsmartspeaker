from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()
import os
import requests
import base64
import numpy as np
import wave
from groq import Groq
from logging_config import logger

app = FastAPI()

# ENV VARS
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Init Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

def pcm8_to_linear16(pcm_data: bytes, sample_rate: int = 16000) -> bytes:
    """
    Converts unsigned 8-bit PCM to signed 16-bit PCM (LINEAR16).
    Args:
        pcm_data: Raw bytes of unsigned 8-bit PCM data
        sample_rate: Sample rate of the audio (default: 16000 Hz)
    Returns:
        bytes: Converted LINEAR16 PCM data
    """
    try:
        # Convert bytes to numpy array of unsigned 8-bit integers
        pcm8 = np.frombuffer(pcm_data, dtype=np.uint8)
        
        # Convert to signed 16-bit PCM (LINEAR16)
        # Formula: (unsigned_8bit - 128) * 256
        linear16 = ((pcm8.astype(np.int16) - 128) * 256).astype(np.int16)
        
        # Convert back to bytes
        return linear16.tobytes()
    except Exception as e:
        logger.error(f"Error in PCM conversion: {str(e)}", exc_info=True)
        raise

def linear16_to_pcm8(audio_data: bytes) -> bytes:
    """
    Converts signed 16-bit PCM (LINEAR16) to unsigned 8-bit PCM.
    Args:
        audio_data: Raw bytes of signed 16-bit PCM data
    Returns:
        bytes: Converted unsigned 8-bit PCM data
    """
    try:
        # Convert bytes to numpy array of signed 16-bit integers
        linear16 = np.frombuffer(audio_data, dtype=np.int16)
        
        # Convert to unsigned 8-bit PCM
        # Formula: (signed_16bit / 256) + 128
        pcm8 = ((linear16 / 256) + 128).astype(np.uint8)
        
        # Convert back to bytes
        return pcm8.tobytes()
    except Exception as e:
        logger.error(f"Error in PCM conversion: {str(e)}", exc_info=True)
        raise

@app.post("/process_audio")
async def process_audio(audio: UploadFile = File(...)):
    try:
        logger.info(f"Received audio file: {audio.filename} with content type: {audio.content_type}")
        
        # Read the audio file
        audio_content = await audio.read()
        file_size = len(audio_content)
        logger.info(f"Audio file size: {file_size} bytes")
        
        # Validate file size (max 10MB)
        if file_size > 10 * 1024 * 1024:
            logger.error(f"File too large: {file_size} bytes")
            return JSONResponse(
                status_code=400,
                content={"error": "Audio file too large. Maximum size is 10MB"}
            )
        
        # Check if the file is PCM/RAW and convert if necessary
        if audio.filename.endswith(('.pcm', '.raw')):
            logger.info("Converting PCM/RAW to LINEAR16 format")
            try:
                audio_content = pcm8_to_linear16(audio_content)
                logger.info("PCM conversion successful")
            except Exception as e:
                logger.error(f"Error in PCM conversion: {str(e)}", exc_info=True)
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Failed to convert PCM audio: {str(e)}"}
                )

        # Convert to base64 for Google STT
        b64_audio = base64.b64encode(audio_content).decode("utf-8")
        logger.info("Audio file converted to base64")

        # 1. Speech to Text (STT)
        logger.info("Sending request to Google Speech-to-Text API")
        try:
            stt_response = requests.post(
                f"https://speech.googleapis.com/v1/speech:recognize?key={GOOGLE_API_KEY}",
                json={
                    "config": {
                        "encoding": "LINEAR16",
                        "languageCode": "en-US",
                        "sampleRateHertz": 16000
                    },
                    "audio": {
                        "content": b64_audio
                    }
                }
            )
            stt_response.raise_for_status()
            stt_result = stt_response.json()
            
            if 'results' not in stt_result or not stt_result['results']:
                logger.error("No speech recognition results returned from Google STT")
                return JSONResponse(
                    status_code=400,
                    content={"error": "No speech recognition results returned"}
                )
                
            user_text = stt_result['results'][0]['alternatives'][0]['transcript']
            logger.info(f"Speech to text conversion successful. User text: {user_text}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Google STT API: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to process speech recognition: {str(e)}"}
            )

        # 2. LLM Response (Groq)
        logger.info("Sending request to Groq LLM")
        try:
            completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a smart speaker assistant. "
                            "Always respond in very short, precise sentences. "
                            "Be clear, friendly, and to-the-point. "
                            "Avoid long explanations. Act like a helpful AI."
                        )
                    },
                    {
                        "role": "user",
                        "content": user_text
                    }
                ],
                model="llama-3.3-70b-versatile",
                stream=False
            )
            llm_reply = completion.choices[0].message.content
            logger.info(f"Received LLM response: {llm_reply}")
        except Exception as e:
            logger.error(f"Error getting LLM response: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to get LLM response: {str(e)}"}
            )

        # 3. Text to Speech (TTS)
        logger.info("Converting text to speech")
        try:
            tts_response = requests.post(
                f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_API_KEY}",
                json={
                    "input": {"text": llm_reply},
                    "voice": {"languageCode": "en-US", "name": "en-US-Wavenet-D"},
                    "audioConfig": {
                        "audioEncoding": "LINEAR16",
                        "sampleRateHertz": 16000
                    }
                }
            )
            tts_response.raise_for_status()
            tts_audio = tts_response.json()['audioContent']
            linear16_bytes = base64.b64decode(tts_audio)
            
            # Convert LINEAR16 to PCM8
            logger.info("Converting LINEAR16 to PCM8 format")
            pcm8_bytes = linear16_to_pcm8(linear16_bytes)
            logger.info("Conversion to PCM8 successful")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Google TTS API: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to convert text to speech: {str(e)}"}
            )

        return StreamingResponse(
            iter([pcm8_bytes]),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": "attachment; filename=response.raw",
                "X-User-Text": user_text,
                "X-LLM-Reply": llm_reply
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error processing audio: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": f"Unexpected error: {str(e)}"}
        )
