# uvicorn be:app --reload --host 0.0.0.0 --port 8001  

import os
import subprocess
import platform
from dotenv import load_dotenv
from groq import Groq
import speech_recognition as sr # Still needed for file format checks, potentially
import paho.mqtt.client as mqtt
import json
import logging
from contextlib import asynccontextmanager # For FastAPI lifespan
import shutil # For saving uploaded file

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware # For PWA interaction
from fastapi.concurrency import run_in_threadpool

# --- Configuration ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEYS")
MQTT_BROKER = os.getenv("MQTT_BROKER", "broker.hivemq.com")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "home/door/control")

TEMP_AUDIO_FILE = "temp_audio_received.wav" # For storing uploaded audio

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Global Clients (to be initialized in lifespan) ---
groq_client = None
mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
mqtt_connected = False

# --- Lifespan Management for FastAPI (startup/shutdown events) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global groq_client, mqtt_connected
    logger.info("FastAPI application startup...")
    if not GROQ_API_KEY:
        logger.error("GROQ_API_KEYS environment variable not set. Groq features will fail.")
        # You might choose to raise an error here and prevent startup
    else:
        try:
            groq_client = Groq(api_key=GROQ_API_KEY)
            logger.info("Groq client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            groq_client = None # Allow app to run, but features will fail

    def on_connect(client, userdata, flags, rc):
        global mqtt_connected
        if rc == 0:
            logger.info(f"Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
            mqtt_connected = True
        else:
            logger.error(f"Failed to connect to MQTT, return code {rc}")
            mqtt_connected = False

    def on_disconnect(client, userdata, rc):
        global mqtt_connected
        logger.info("Disconnected from MQTT broker.")
        mqtt_connected = False
        # Optional: implement reconnection logic here if desired

    mqtt_client.on_connect = on_connect
    mqtt_client.on_disconnect = on_disconnect

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start() # Start a background thread for MQTT
    except Exception as e:
        logger.error(f"Failed to connect to MQTT broker on startup: {e}")

    yield # API is running

    # Shutdown
    logger.info("FastAPI application shutdown...")
    if mqtt_client.is_connected():
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
    logger.info("MQTT client disconnected.")
    if os.path.exists(TEMP_AUDIO_FILE):
        try:
            os.remove(TEMP_AUDIO_FILE)
            logger.info(f"Cleaned up temporary file: {TEMP_AUDIO_FILE}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary file {TEMP_AUDIO_FILE}: {e}")


app = FastAPI(lifespan=lifespan)

# CORS (Cross-Origin Resource Sharing) middleware
# Allows your React PWA (running on a different port/domain) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # TEMPORARY: Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Core Functions (Adapted for API) ---
async def transcribe_audio_from_file(audio_file_path: str):
    if not groq_client:
        logger.error("Groq client not initialized during transcription attempt.")
        raise HTTPException(status_code=500, detail="Transcription service not available (Groq client error).")
    logger.info(f"Transcribing audio from file: {audio_file_path}")
    try:
        with open(audio_file_path, "rb") as audio_file_content: # Renamed to avoid conflict
            # Run the blocking Groq call in a thread pool
            transcription = await run_in_threadpool(
                groq_client.audio.transcriptions.create,
                file=(os.path.basename(audio_file_path), audio_file_content.read()),
                model="whisper-large-v3",
                response_format="verbose_json"
            )
        logger.info(f"Transcription result: {transcription.text}")
        if not transcription.text.strip():
            logger.warning("Transcription resulted in empty text.")
            # Consider this an error or handle as "no speech"
            raise ValueError("Transcription resulted in empty text. No speech detected or understood.")
        return transcription.text
    except ValueError as ve: # Catch our specific empty transcription error
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


async def get_llm_function_call(text: str):
    if not groq_client:
        logger.error("Groq client not initialized during LLM call attempt.")
        raise HTTPException(status_code=500, detail="LLM service not available (Groq client error).")
    logger.info(f"Performing LLM function call for text: {text}")
    prompt = f"""
You are a smart assistant. When the user gives a command, respond ONLY with a JSON object that matches one of the function schemas.

Available functions:
- "open_door": Opens the front door. No parameters.
- "close_door": Closes the front door. No parameters.

Examples:
User: "Please open the door"
Output: {{ "function": "open_door" }}

User: "Can you close the door?"
Output: {{ "function": "close_door" }}

Now process this:
User: "{text}"
Output:
"""
    try:
        # Run the blocking Groq call in a thread pool
        response = await run_in_threadpool(
            groq_client.chat.completions.create,
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
            stream=False,
        )
        result_text = response.choices[0].message.content.strip()
        logger.info(f"LLM function call raw result: {result_text}")
        return result_text
    except Exception as e:
        logger.error(f"Error during LLM function call: {e}")
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")

def send_mqtt_command_from_llm(json_command_str: str):
    global mqtt_connected
    if not mqtt_connected:
        logger.error("MQTT client not connected. Cannot send command.")
        # Attempt a one-off reconnect, or rely on the background loop to reconnect
        if not mqtt_client.is_connected():
            try:
                logger.info("Attempting MQTT reconnect before publish...")
                mqtt_client.reconnect() # This will block briefly
                # A short wait might be needed for on_connect to fire and update mqtt_connected
                # but for a single command, it's tricky. Best to ensure it's usually connected.
            except Exception as e:
                logger.error(f"MQTT reconnect attempt failed: {e}")
                return False, "MQTT client not connected. Command not sent."
        if not mqtt_client.is_connected(): # Check again
             return False, "MQTT client still not connected after reconnect attempt."


    logger.info(f"Attempting to send MQTT command: {json_command_str}")
    try:
        if json_command_str.startswith("```json"):
            json_command_str = json_command_str.replace("```json", "").replace("```", "").strip()

        payload = json.loads(json_command_str)
        command_function = payload.get("function")

        mqtt_payload_dict = None
        if command_function == "open_door":
            mqtt_payload_dict = {"command": "open", "device": "front_door"}
        elif command_function == "close_door":
            mqtt_payload_dict = {"command": "close", "device": "front_door"}
        else:
            logger.warning(f"Unknown command function from LLM: {command_function}")
            return False, f"Unknown command: '{command_function}'."

        if mqtt_payload_dict:
            mqtt_payload = json.dumps(mqtt_payload_dict)
            result = mqtt_client.publish(MQTT_TOPIC, mqtt_payload)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Published to {MQTT_TOPIC}: {mqtt_payload}. MID: {result.mid}")
                return True, f"{command_function.replace('_',' ').capitalize()} command sent."
            else:
                logger.error(f"Failed to publish MQTT message. RC: {result.rc}")
                return False, f"Failed to send MQTT command (RC: {result.rc})."

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON response: {e}. Response was: '{json_command_str}'")
        return False, f"Error: Invalid format from assistant: {json_command_str}"
    except Exception as e:
        logger.error(f"Failed to send MQTT message: {e}")
        return False, f"Error sending MQTT command: {str(e)}"
    return False, "No valid MQTT command processed." # Should not be reached if logic is correct

# --- API Endpoint ---
@app.post("/process-command/")
async def process_command_endpoint(audio_file: UploadFile = File(...)):
    if not groq_client:
        raise HTTPException(status_code=503, detail="Groq client not initialized. Service unavailable.")
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    # Save uploaded file temporarily
    try:
        with open(TEMP_AUDIO_FILE, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        logger.info(f"Received audio file: {audio_file.filename}, saved as {TEMP_AUDIO_FILE}")
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Could not save uploaded audio file.")
    finally:
        audio_file.file.close() # Important to close the file stream

    transcribed_text = ""
    llm_result_json = ""
    mqtt_success = False
    mqtt_message = "Processing not completed."

    try:
        # 1. Transcribe Audio
        transcribed_text = await transcribe_audio_from_file(TEMP_AUDIO_FILE)
        if not transcribed_text: # Should be caught by transcribe_audio_from_file, but double check
            raise HTTPException(status_code=400, detail="Transcription resulted in no text.")

        # 2. Get LLM Function Call
        llm_result_json = await get_llm_function_call(transcribed_text)

        # 3. Send MQTT Command
        if not mqtt_client or not mqtt_client.is_connected():
            logger.warning("MQTT client not connected when trying to send command.")
            # You might want to attempt a reconnect here or just fail
            raise HTTPException(status_code=503, detail="MQTT service unavailable. Cannot send command.")

        mqtt_success, mqtt_message = send_mqtt_command_from_llm(llm_result_json)

        return JSONResponse(content={
            "status": "success" if mqtt_success else "partial_failure",
            "transcribed_text": transcribed_text,
            "llm_command": llm_result_json,
            "mqtt_status": mqtt_message,
            "mqtt_sent_successfully": mqtt_success
        })

    except HTTPException as http_exc: # Re-raise FastAPI's HTTPExceptions
        logger.error(f"HTTP Exception in processing: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Unhandled error in process_command_endpoint: {e}")
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e),
            "transcribed_text": transcribed_text, # Include intermediate results if available
            "llm_command": llm_result_json
        })
    finally:
        # Clean up the temporary audio file
        if os.path.exists(TEMP_AUDIO_FILE):
            try:
                os.remove(TEMP_AUDIO_FILE)
                logger.info(f"Cleaned up {TEMP_AUDIO_FILE} after processing.")
            except Exception as e:
                logger.error(f"Error deleting temp file {TEMP_AUDIO_FILE}: {e}")


if __name__ == "__main__":
    import uvicorn
    # To run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
    # The host 0.0.0.0 makes it accessible on your network
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
