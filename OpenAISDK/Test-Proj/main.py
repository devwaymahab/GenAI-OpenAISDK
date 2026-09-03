from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

client = OpenAI()
response = client.responses.create(
    model="gpt-4.1-mini",
    input="Write a short poem about the beauty of nature."
)
print(response)