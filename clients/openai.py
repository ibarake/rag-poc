from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

openai_api = OpenAI()
openai_api.api_key = os.getenv("OPENAI_API_KEY")