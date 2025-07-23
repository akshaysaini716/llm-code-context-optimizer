import google.generativeai as genai
import os
import tiktoken
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure with API key from environment
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.0-flash")

# Initialize tiktoken encoder (using GPT-4 encoding as a reasonable approximation)
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken"""
    return len(encoding.encode(text))

def chat_with_gemini(prompt: str):
    chat = model.start_chat()
    response = chat.send_message(prompt)
    
    # Count tokens using tiktoken
    prompt_tokens = count_tokens(prompt)
    completion_tokens = count_tokens(response.text)
    total_tokens = prompt_tokens + completion_tokens
    
    return {
        "response": response.text,
        "tokens_used": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    }
