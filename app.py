from flask import Flask, request, jsonify
import whisper
import os
import tempfile
import logging
import time
import threading

app = Flask(__name__)

ALLOWED_MODELS = {"tiny.en", "tiny", "base.en", "base", "small.en", "small", "medium.en", "medium", "large", "large-v1", "large-v2", "large-v3"}
DEFAULT_MODEL = "base"

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(name)s:%(process)d:%(message)s')
logger = logging.getLogger(__name__)

loaded_models = {}
model_lock = threading.Lock()

def load_whisper_model(model_name): # Removed default here, validation happens before call
    """Loads or retrieves a Whisper model from the cache. Assumes model_name is validated."""
    global loaded_models
    global model_lock

    with model_lock:
        if model_name in loaded_models:
            logger.info(f"Using cached model: {model_name}")
            return loaded_models[model_name]

        try:
            logger.info(f"Attempting to load Whisper model: {model_name}...")
            start_time = time.time()
            model_instance = whisper.load_model(model_name)
            load_time = time.time() - start_time
            logger.info(f"Whisper model '{model_name}' loaded successfully in {load_time:.2f} seconds.")
            loaded_models[model_name] = model_instance # Store in cache
            return model_instance
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{model_name}': {e}", exc_info=True)
            return None

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    """Handles audio file uploads and returns the transcription using a requested model."""

    requested_model_name = request.form.get('model_name', DEFAULT_MODEL)
    logger.info(f"Received transcription request, initially asking for model: '{requested_model_name}'")

    if requested_model_name not in ALLOWED_MODELS:
        logger.warning(f"Requested model '{requested_model_name}' is not in ALLOWED_MODELS. Falling back to default '{DEFAULT_MODEL}'.")
        actual_model_name = DEFAULT_MODEL
    else:
        actual_model_name = requested_model_name

    logger.info(f"Attempting to use model: '{actual_model_name}'")

    model = load_whisper_model(actual_model_name) # Pass the validated name

    if model is None:
        return jsonify({"error": f"Whisper model '{actual_model_name}' is not available or failed to load."}), 503 # Service Unavailable

    if 'audio_file' not in request.files:
        logger.warning("Request received without 'audio_file' part.")
        return jsonify({"error": "No audio file part in the request"}), 400

    audio_file = request.files['audio_file']

    if audio_file.filename == '':
        logger.warning("Request received with an empty filename for the audio file.")
        return jsonify({"error": "No selected audio file (empty filename)"}), 400

    temp_audio_path = None
    try:
        _, suffix = os.path.splitext(audio_file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or ".tmp") as temp_audio:
            audio_file.save(temp_audio)
            temp_audio_path = temp_audio.name

        logger.info(f"Temporary audio file saved to: {temp_audio_path}")

        logger.info(f"Starting transcription with model '{actual_model_name}' for: {temp_audio_path}")
        start_time = time.time()
        result = model.transcribe(temp_audio_path)
        transcription_time = time.time() - start_time
        logger.info(f"Transcription completed in {transcription_time:.2f} seconds for: {temp_audio_path}")

        return jsonify({
            "text": result["text"],
            "model_used": actual_model_name
            })

    except Exception as e:
        logger.error(f"An error occurred during transcription: {e}", exc_info=True)
        return jsonify({"error": f"An internal error occurred during transcription: {str(e)}"}), 500

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
                logger.info(f"Removed temporary file: {temp_audio_path}")
            except OSError as e:
                logger.error(f"Error removing temporary file {temp_audio_path}: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200
