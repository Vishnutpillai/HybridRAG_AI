import os

from dotenv import load_dotenv
from groq import Groq


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

# ============================================================
# READ GROQ API KEY
# ============================================================

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "❌ GROQ_API_KEY not found! "
        "Please add it to your .env file."
    )

# ============================================================
# CREATE GROQ CLIENT
# ============================================================

client = Groq(
    api_key=api_key
)

# ============================================================
# ASK GROQ
# ============================================================

def ask_groq(prompt: str) -> str:
    """
    Send a prompt to Groq LLM and return the response.
    """

    if not prompt.strip():
        raise ValueError(
            "❌ Prompt cannot be empty!"
        )

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

# ============================================================
# TEST GROQ
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("GROQ LLM TEST")
    print("=" * 60)

    question = "What is machine learning?"

    print(f"\n❓ Question: {question}")

    try:

        answer = ask_groq(question)

        print("\n🤖 Groq Response:")
        print("-" * 60)
        print(answer)

        print("\n" + "=" * 60)
        print("✅ GROQ TEST PASSED")
        print("=" * 60)

    except Exception as e:

        print("\n" + "=" * 60)
        print("❌ GROQ TEST FAILED")
        print("=" * 60)

        print(f"\nError: {str(e)}")

        raise

