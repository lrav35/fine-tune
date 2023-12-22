from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-tiny"

client = MistralClient(api_key=api_key)

messages = [
	ChatMessage(role="user", content="translate this sentence to french: I would like a coffee, but please go light on the milk.")
    ]

chat_response = client.chat(
	model=model,
	messages=messages,
)

print(chat_response.choices[0].message.content)
