from dotenv import load_dotenv
import os
import tiktoken
import google.generativeai as genai

class GeminiService:
    def __init__(self, model_name="gemini-2.0-flash"):
        load_dotenv()
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name)
        # Initialize tiktoken encoder for token count approximation
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def chat_with_gemini(self, prompt: str):
        chat = self.model.start_chat()
        response = chat.send_message(prompt)

        # Count tokens using tiktoken
        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = self.count_tokens(response.text)
        total_tokens = prompt_tokens + completion_tokens

        return {
            "response": response.text,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            }
        }

