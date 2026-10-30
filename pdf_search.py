from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

load_dotenv() 
client = OpenAI()

response = client.responses.create( 
                model="gpt-4.1-mini", 
                input="What did the Queen ask the magic mirror? Return ONLY the exact answer to the question.",
                tools = [ 
                         { "type": "file_search",
                          "vector_store_ids":["vs_6ac535e7e4c481919f5498bad7598957"]
                          }]
                )

print(response.output_text)