import os

from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Read API key
api_key = os.getenv("GROQ_API_KEY")

# Create Groq client
client = Groq(api_key=api_key)


def ask_groq(prompt: str) -> str:
    """Send a prompt to the Groq LLM and return the response."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        max_tokens=512
    )

    return response.choices[0].message.content