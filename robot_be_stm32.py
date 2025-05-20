# uvicorn robot_be_stm32:app --reload --host 0.0.0.0 --port 8001  

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
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
# NEW/UPDATED MQTT TOPIC FOR ROBOT
MQTT_ROBOT_TOPIC = "esp8266/robot/command_json"

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
    else:
        try:
            groq_client = Groq(api_key=GROQ_API_KEY)
            logger.info("Groq client initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            groq_client = None

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

    mqtt_client.on_connect = on_connect
    mqtt_client.on_disconnect = on_disconnect

    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        logger.error(f"Failed to connect to MQTT broker on startup: {e}")

    yield

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        with open(audio_file_path, "rb") as audio_file_content:
            transcription = await run_in_threadpool(
                groq_client.audio.transcriptions.create,
                file=(os.path.basename(audio_file_path), audio_file_content.read()),
                model="whisper-large-v3",
                response_format="verbose_json"
            )
        logger.info(f"Transcription result: {transcription.text}")
        if not transcription.text.strip():
            logger.warning("Transcription resulted in empty text.")
            raise ValueError("Transcription resulted in empty text. No speech detected or understood.")
        return transcription.text
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


async def get_llm_function_call_for_robot(text: str): # Renamed for clarity
    if not groq_client:
        logger.error("Groq client not initialized during LLM call attempt.")
        raise HTTPException(status_code=500, detail="LLM service not available (Groq client error).")
    logger.info(f"Performing LLM function call for robot command: {text}")
    prompt = f"""
You are an assistant controlling a robot car. When the user gives a command, respond ONLY with a JSON object that matches one of the function schemas for robot control.

Available robot functions:
- "move_forward": Moves the robot forward. No parameters.
- "move_backward": Moves the robot backward. No parameters.
- "turn_left": Turns the robot left. No parameters.
- "turn_right": Turns the robot right. No parameters.
- "stop_robot": Stops the robot. No parameters.

Examples:
User: "Robot, go forward."
Output: {{ "function": "move_forward" }}

User: "Make the car turn left."
Output: {{ "function": "turn_left" }}

User: "Stop moving."
Output: {{ "function": "stop_robot" }}

User: "Reverse the vehicle."
Output: {{ "function": "move_backward" }}

Now process this:
User: "{text}"
Output:
"""
    try:
        response = await run_in_threadpool(
            groq_client.chat.completions.create,
            model="llama-3.1-8b-instant", # Or your preferred model
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=100,
            stream=False,
        )
        result_text = response.choices[0].message.content.strip()
        logger.info(f"Robot LLM function call raw result: {result_text}")
        return result_text
    except Exception as e:
        logger.error(f"Error during LLM function call for robot: {e}")
        raise HTTPException(status_code=500, detail=f"LLM processing failed: {str(e)}")

def send_robot_command_via_mqtt(llm_json_output_str: str): # Parameter name changed for clarity
    global mqtt_connected
    if not mqtt_connected:
        logger.error("MQTT client not connected. Cannot send robot command.")
        if not mqtt_client.is_connected():
            try:
                logger.info("Attempting MQTT reconnect before publish...")
                mqtt_client.reconnect()
            except Exception as e:
                logger.error(f"MQTT reconnect attempt failed: {e}")
                return False, "MQTT client not connected. Robot command not sent."
        if not mqtt_client.is_connected():
             return False, "MQTT client still not connected after reconnect attempt for robot."

    logger.info(f"Processing LLM output for MQTT: {llm_json_output_str}")
    try:
        # Clean up potential markdown code block formatting from LLM
        if llm_json_output_str.startswith("```json"):
            llm_json_output_str = llm_json_output_str.replace("```json", "").replace("```", "").strip()

        # Parse the JSON from the LLM
        llm_payload_data = json.loads(llm_json_output_str)
        llm_function_name = llm_payload_data.get("function")

        # Translate LLM function name to the command string expected by ESP8266's executeCommand
        # This is the value that will go into the "function" field of the JSON sent to ESP8266
        esp_command_value = None
        if llm_function_name == "move_forward":
            esp_command_value = "forward"
        elif llm_function_name == "move_backward":
            esp_command_value = "backward"
        elif llm_function_name == "turn_left":
            esp_command_value = "turnleft"
        elif llm_function_name == "turn_right":
            esp_command_value = "turnright"
        elif llm_function_name == "stop_robot":
            esp_command_value = "stop"
        else:
            logger.warning(f"Unknown robot command function from LLM: {llm_function_name}")
            return False, f"Unknown robot command: '{llm_function_name}'."

        if esp_command_value:
            # Construct the JSON payload for the ESP8266
            # The ESP8266 expects a JSON like: {"function": "actual_command_for_stm32"}
            mqtt_payload_for_esp = {"function": esp_command_value}
            mqtt_payload_json_string = json.dumps(mqtt_payload_for_esp)

            # Publish the constructed JSON string
            result = mqtt_client.publish(MQTT_ROBOT_TOPIC, mqtt_payload_json_string)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Published to {MQTT_ROBOT_TOPIC}: '{mqtt_payload_json_string}'. MID: {result.mid}")
                return True, f"Robot command '{esp_command_value}' (sent as JSON) published."
            else:
                logger.error(f"Failed to publish MQTT message for robot. RC: {result.rc}")
                return False, f"Failed to send robot MQTT command (RC: {result.rc})."
        else: # Should be caught by the unknown command check above, but as a safeguard
            return False, "No valid ESP command value derived from LLM output."


    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON response for robot: {e}. Response was: '{llm_json_output_str}'")
        return False, f"Error: Invalid format from assistant for robot command: {llm_json_output_str}"
    except Exception as e:
        logger.error(f"Failed to send MQTT message for robot: {e}")
        return False, f"Error sending robot MQTT command: {str(e)}"
    return False, "No valid robot MQTT command processed." # Fallback

# --- API Endpoint ---
@app.post("/process-command/") # Changed endpoint name for clarity
async def process_robot_command_endpoint(audio_file: UploadFile = File(...)):
    if not groq_client:
        raise HTTPException(status_code=503, detail="Groq client not initialized. Service unavailable.")
    if not audio_file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")

    try:
        with open(TEMP_AUDIO_FILE, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        logger.info(f"Received audio file for robot: {audio_file.filename}, saved as {TEMP_AUDIO_FILE}")
    except Exception as e:
        logger.error(f"Error saving uploaded file for robot: {e}")
        raise HTTPException(status_code=500, detail="Could not save uploaded audio file.")
    finally:
        audio_file.file.close()

    transcribed_text = ""
    llm_result_json_str = "" # Store the raw JSON string from LLM
    mqtt_success = False
    mqtt_message = "Processing not completed."

    try:
        transcribed_text = await transcribe_audio_from_file(TEMP_AUDIO_FILE)
        if not transcribed_text:
            raise HTTPException(status_code=400, detail="Transcription resulted in no text.")

        llm_result_json_str = await get_llm_function_call_for_robot(transcribed_text)

        if not mqtt_client or not mqtt_client.is_connected():
            logger.warning("MQTT client not connected when trying to send robot command.")
            raise HTTPException(status_code=503, detail="MQTT service unavailable. Cannot send robot command.")

        # Pass the raw LLM JSON output string to the MQTT sending function
        mqtt_success, mqtt_message = send_robot_command_via_mqtt(llm_result_json_str)

        return JSONResponse(content={
            "status": "success" if mqtt_success else "partial_failure",
            "transcribed_text": transcribed_text,
            "llm_command_details": llm_result_json_str, # Return the raw LLM JSON
            "mqtt_status": mqtt_message,
            "mqtt_sent_successfully": mqtt_success
        })

    except HTTPException as http_exc:
        logger.error(f"HTTP Exception in robot processing: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Unhandled error in process_robot_command_endpoint: {e}")
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": str(e),
            "transcribed_text": transcribed_text,
            "llm_command_details": llm_result_json_str # Return LLM JSON even on error
        })
    finally:
        if os.path.exists(TEMP_AUDIO_FILE):
            try:
                os.remove(TEMP_AUDIO_FILE)
                logger.info(f"Cleaned up {TEMP_AUDIO_FILE} after robot processing.")
            except Exception as e:
                logger.error(f"Error deleting temp file {TEMP_AUDIO_FILE}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("robot_be_stm32:app", host="0.0.0.0", port=8001, reload=True) # Pass app as string for reload