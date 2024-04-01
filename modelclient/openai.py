from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
openai_api = OpenAI()
openai_api.api_key = os.getenv("OPENAI_API_KEY")