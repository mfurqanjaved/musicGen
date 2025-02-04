#define libraries
from fastapi import FastAPI, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import pipeline
import torch
import os
from dotenv import load_dotenv
import replicate
import httpx
import json
import asyncio

app = FastAPI()

# Load environment variables
load_dotenv()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load HTML templates
templates = Jinja2Templates(directory="templates")

# Function to generate lyrics using Hugging Face's GPT-NEO model
def generate_lyrics(prompt):
    # Initialize text generation pipeline with GPT-NEO model
    generator = pipeline('text-generation', model='EleutherAI/gpt-neo-1.3B')
    # Generate lyrics based on the prompt
    response = generator(prompt, max_length=50, temperature=0.7, do_sample=True)
    # Extract generated text from response
    output = response[0]['generated_text']
    # Format the generated lyrics
    cleaned_output = output.replace("\n", " ")
    formatted_lyrics = f"♪ {cleaned_output} ♪"
    return formatted_lyrics

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate-music")
async def generate_music(prompt: str = Form(...), duration: int = Form(...)):
    try:
        lyrics = generate_lyrics(prompt)
        prompt_with_lyrics = lyrics
        print(f"Generated lyrics: {prompt_with_lyrics}")
        
        # Increased timeout and adjusted retry settings
        max_retries = 5  # Increased from 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                output = replicate.run(
                    "suno-ai/bark:b76242b40d67c76ab6742e987628a2a9ac019e11d56ab96c4e91ce03b79b2787",
                    input={
                        "prompt": prompt_with_lyrics,
                        "text_temp": 0.7,
                        "output_full": False,
                        "waveform_temp": 0.7
                    },
                    timeout=600  # Increased from 300 to 600 seconds (10 minutes)
                )
                if output:  # Check if we got a valid response
                    break
                retry_count += 1
            except (replicate.exceptions.ReplicateError, httpx.ReadTimeout) as e:
                retry_count += 1
                print(f"API error (attempt {retry_count}/{max_retries}): {str(e)}")
                if retry_count == max_retries:
                    return JSONResponse(
                        status_code=504,
                        content={"error": "Service timeout. Please try again later."}
                    )
                await asyncio.sleep(min(2 ** retry_count, 30))  # Cap max wait at 30 seconds
        
        print(f"Raw output type: {type(output)}")
        print(f"Raw output content: {output}")
        
        if isinstance(output, dict) and 'audio_out' in output:
            print("Output is valid dictionary with audio_out key")
            audio_out = output['audio_out']
            print(f"Audio out object type: {type(audio_out)}")
            
            # Convert FileOutput to string immediately
            music_url = str(audio_out)
            print(f"Converted URL: {music_url}")
            
            # Ensure we have a valid URL string
            if not music_url or not isinstance(music_url, str):
                raise ValueError("Invalid URL generated")
            
            safe_response = {"url": music_url}
            
            # Verify the response is JSON serializable
            try:
                json.dumps(safe_response)
            except TypeError as e:
                print(f"Serialization test failed: {e}")
                raise ValueError("Response is not JSON serializable")
                
            return JSONResponse(
                content=safe_response,
                media_type="application/json"
            )
        else:
            error_msg = f"Unexpected output format: {output}"
            print(error_msg)
            return JSONResponse(
                status_code=500,
                content={"error": error_msg}
            )
    except Exception as e:
        error_msg = f"Error processing output: {str(e)}"
        print(error_msg)
        return JSONResponse(
            status_code=500,
            content={"error": error_msg}
        )