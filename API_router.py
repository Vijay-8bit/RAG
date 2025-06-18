from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
import requests
from Prompt import retrieve_context
import json
import os
from dotenv import load_dotenv
load_dotenv()

router = APIRouter()
templates = Jinja2Templates(directory= "static")

OLLAMA_URL = os.getenv("OLLAMA_URL")
MODEL_NAME = os.getenv("OLLAMA_CHAT_MODEL")

@router.get('/', response_class= HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("a.html", {"request": request})

@router.post('/send_message')
async def stream_response(request: Request):
    data = await request.json()
    user_input = data.get("message","")    

    def event_stream():

        with requests.post(
            OLLAMA_URL, 
            json = {
                "model": MODEL_NAME,
                "stream":True,
                "messages":[
                    {"role": "user", "content": retrieve_context(user_input)}
                ]
            },
            stream = True
        ) as r:
            result = ''
            for line in r.iter_lines():
                if line: 
                    json_data= json.loads(line.decode("utf-8"))
                    content = json_data.get("message", {}).get("content", "")
                    if content:
                        result += content
                        # print("DEBUG: Streaming content:", result)
                        yield f"data: {content}\n\n"
    
                        
    return StreamingResponse(event_stream(), media_type = 'text/event-stream')


