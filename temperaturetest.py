from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

client = OpenAI()

prompt = "Give me a creative startup idea in one sentence."
temperature = 0.2

response = client.responses.create(
    model="gpt-4.1-mini",
    input=prompt,
    temperature = temperature
        
)
print("="*10+str(temperature)+"="*10)
print(response.output_text)