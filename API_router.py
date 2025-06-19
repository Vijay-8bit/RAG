import logging
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import requests
from Prompt import retrieve_context
import json
import os
from dotenv import load_dotenv
from langfuse import get_client
from langfuse_config import langfuse
from datetime import datetime

load_dotenv()

router = APIRouter()
templates = Jinja2Templates(directory="static")

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL_NAME = os.getenv("OLLAMA_CHAT_MODEL")

# Configure logging
logging.basicConfig(
    filename="app_logs.log",  # Log file name
    level=logging.DEBUG,      # Log level
    format="%(asctime)s - %(levelname)s - %(message)s"  # Log format
)

input_output_store = []

@router.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("a.html", {"request": request})


@router.post('/ping')
async def handle_ping(request: Request):
    """Handles ping requests from the frontend."""
    data = await request.json()
    user_input = data.get("message", "")
    logging.info(f"Received ping request with input: {user_input}")  # Log ping request

    # Check if the input is "ping"
    if user_input.lower() == "ping":
        logging.info("Ping request successfully handled")  # Log successful ping handling
        return JSONResponse(content={"status": "ok", "message": "Ping received successfully"})
    else:
        logging.warning(f"Unexpected input in ping endpoint: {user_input}")  # Log unexpected input
        return JSONResponse(content={"status": "error", "message": "Invalid ping request"})


@router.post('/send_message')
async def stream_response(request: Request):
    data = await request.json()
    user_input = data.get("message", "")

    logging.info(f"Received user input api: {user_input}")  # Log user input

    # Start Langfuse parent span
    request_span = langfuse.start_span(
        name="chat-request",
        metadata={
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input
        }
    ) # Log span creation

    def event_stream():
        try:
            # Start context retrieval span
            context_span = request_span.start_span(name="context-retrieval")
            # Log span creation

            prompt = retrieve_context(user_input)
            

            context_span.update(
                metadata={"generated_prompt": prompt}
            )
            logging.info(f"Retrieved data IN api file is: {prompt}")
            context_span.end()

            # Start LLM interaction span
            llm_span = request_span.start_span(name="llm-interaction")
            
            with requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "stream": True,
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_input}
                    ]
                },
                stream=True
            ) as r:
                result = ''
                for line in r.iter_lines():
                    if line:
                        json_data = json.loads(line.decode("utf-8"))
                        content = json_data.get("message", {}).get("content", "")
                        if content:
                            result += content
                            logging.debug(f"Streaming content: {content}")  # Log streamed content
                            yield f"data: {content}\n\n"

                llm_span.update(
                    metadata={
                        "model": MODEL_NAME,
                        "complete_response": result
                    }
                )
                
                llm_span.end()

              

        finally:
            # End parent span
            request_span.update(
                metadata={"status": "completed"}
            )
            
            request_span.end()
            langfuse.flush()

    return StreamingResponse(event_stream(), media_type='text/event-stream')