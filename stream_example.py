from openai import OpenAI
from dotenv import load_dotenv
import time
load_dotenv()  # Load environment variables from .env file

client = OpenAI()

prompt = "Explain Python decorators in simple terms"
print("\n--WITH STREAMING----")
start = time.perf_counter()
response = client.responses.create(
    model="gpt-4.1-mini",
    input=prompt,
    stream = True
        
)
first_token_time = None
for event in response:
    if event.type == 'response.output_text.delta':
        if first_token_time is None:
            first_token_time = time.perf_counter()
            print(f"\nfirst output after:{first_token_time-start:.2f}")
        print(event.delta,end="",flush=True)
        
end = time.perf_counter()
        
print(f"\n Time taken={end-start:.2f}")






