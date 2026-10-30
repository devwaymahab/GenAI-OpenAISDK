from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from pydantic import BaseModel
class User(BaseModel):
    name: str
    age: int
    email: str
    skills: list[str]


client = OpenAI()
response = client.responses.parse(
    model="gpt-4.1-mini",
    input="Rahul is 28 years old. His email is rahul@gmail.com. "
          "He knows Python, Django and FastAPI.",
          
    text_format = User
)
print(response.output_text)
print(type(response.output_text))
print(response.output_parsed)
print(type(response.output_parsed))
print(response.output_parsed.name)
print(response.output_parsed.email)
print(response.output_parsed.skills)