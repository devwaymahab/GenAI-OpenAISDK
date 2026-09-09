from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

load_dotenv()  # Load environment variables from .env file

client = OpenAI()


def get_current_time(timezone):
    now = datetime.now(ZoneInfo(timezone))
    return {
        "date":now.strftime("%Y-%m-%d"),
        "time":now.strftime("%I:%M:%p"),
        "timezone":timezone
    }


tools = [
    {
        "type":"function",
        "name":"get_current_time",
        "description":"get the current date and time for a timezone",
        "parameters":{
            "type":"object",
            "properties":{
                "timezone":{
                    "type":"string",
                    "description":"IANA timezone, for example Asia/Kolkata"
                }
            },
            "required":["timezone"]
        }
    }
]

response = client.responses.create(
    model="gpt-4.1-mini",
    input="What is current date and time in IST",
    tools = tools
)
print(response.output)

available_tools = {
    "get_current_time":get_current_time
}
import json
for item in response.output:
    if item.type=="function_call":
        tool = available_tools[item.name]
        arguments = json.loads(item.arguments)
        
        result = tool(**arguments)
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": json.dumps(result)
                }
            ],
            previous_response_id=response.id
        )
        print(response.output_text)