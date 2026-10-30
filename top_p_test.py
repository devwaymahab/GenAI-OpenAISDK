from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

client = OpenAI()

prompt = "Give me 5 different names for a food delivery app. \
    Return only the names, one per line."
top_p = 1
response = client.responses.create(
    model="gpt-4.1-mini",
    input=prompt,
    top_p = top_p
        
)
print("="*10+str(top_p)+"="*10)
print(response.output_text)