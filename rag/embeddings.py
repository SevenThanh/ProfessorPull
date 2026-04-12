import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
def embed(text, task="document"):
    t = f"RETRIEVAL_{'DOCUMENT' if task == 'document' else 'QUERY'}"
    res = genai.embed_content(model="models/gemini-embedding-001", content=text, task_type=t)
    return res["embedding"]