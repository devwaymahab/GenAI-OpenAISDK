from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() 
client = OpenAI()

# vector_store = client.vector_stores.create(
#     name="company_docs"
# )

# print(vector_store.id)
# file = client.files.create(
#     file=open("manual.pdf", "rb"),
#     purpose="assistants"
# )

# client.vector_stores.files.create(
#     vector_store_id=vector_store.id,
#     file_id=file.id
# )

#######################
response = client.responses.create(
    model="gpt-4.1-mini",
    input="What the Mirror answer to everyday? give only answer to this",
    tools=[
        {
            "type": "file_search",
            "vector_store_ids": ["vs_6aa3007ad99881918b95859791103694"]
        }
    ]
)

print(response.output_text)