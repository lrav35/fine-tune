from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-tiny"

client = MistralClient(api_key=api_key)

messages = [
	ChatMessage(role="user", content="can you translate this sentence to french for me? sentence: I would like a coffee with milk, but go light on the sugar.")
    ]

chat_response = client.chat(
	model=model,
	messages=messages,
)

print(chat_response.choices[0].message.content)
