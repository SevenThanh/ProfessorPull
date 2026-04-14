from transformers import pipeline

model1 = pipeline("text-generation", model="Qwen/Qwen2.5-Coder-32B-Instruct")
messages = [
    {"role": "user", "content": "Who are you?"},
]
model1(messages)

model2 = pipeline("text-generation", model="deepseek-ai/DeepSeek-V3.2")
messages = [
    {"role": "user", "content": "Who are you?"},
]
model2(messages)

model3 = pipeline("text-generation", model="meta-llama/Llama-3.3-70B-Instruct")
messages = [
    {"role": "user", "content": "Who are you?"},
]
model3(messages)