import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from rich.console import Console
from rich.markdown import Markdown

console=Console()

# Load Gemma model and tokenizer
model_name = "google/gemma-3-1b-it"
device="cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name,token=True)
model = AutoModelForCausalLM.from_pretrained(model_name, token=True)
model.to(device)

def chat_with_gemma():
    print("Gemma 3 AI Chatbot (type 'exit' to quit)")
    conversation_history = []

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Format input as a conversation
        conversation_history.append({"role": "user", "content": user_input})
        prompt = tokenizer.apply_chat_template(conversation_history, tokenize=False, add_generation_prompt=True)

        # Generate response
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_length=1024, pad_token_id=tokenizer.eos_token_id, do_sample=True)
        response = tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract bot response and update history
        bot_reply = response[len(re.sub("<.*>","",prompt)):]
        conversation_history.append({"role": "assistant", "content": bot_reply.strip()})

        console.print(Markdown("Bot: "+bot_reply))

if __name__ == "__main__":
    chat_with_gemma()
