import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("Missing env var OPENAI_API_KEY")

_oai = OpenAI()


def embed(text, task="document"):
    r = _oai.embeddings.create(model="text-embedding-3-small", input=text, dimensions=768)
    return list(r.data[0].embedding)
