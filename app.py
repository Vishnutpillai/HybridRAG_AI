from src.groq_con import ask_groq

question = "What is Retrieval-Augmented Generation (RAG)?"

answer = ask_groq(question)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(answer)