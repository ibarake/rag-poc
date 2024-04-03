from openai import OpenAI
import os

openai_api = OpenAI()
openai_api.api_key = os.getenv("OPENAI_API_KEY")