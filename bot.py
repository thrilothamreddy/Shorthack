from openai import OpenAI
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
client = OpenAI()  # ✅ new client object (automatically uses OPENAI_API_KEY)

def ask_ai(question):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if allowed by your org
            messages=[
                {"role": "system", "content": "You are a helpful assistant that supports rent estimation."},
                {"role": "user", "content": question}
            ]
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Error: {str(e)}"
