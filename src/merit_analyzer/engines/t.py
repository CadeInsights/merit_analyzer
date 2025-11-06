from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex


c = AnthropicBedrock()
message = c.messages.create(
    max_tokens=1024,
    messages=[
        {
            "role": "user",
            "content": "Hello!",
        }
    ],
    model="anthropic.claude-sonnet-4-5-20250929-v1:0",
)
print(message.model_dump_json(indent=2))