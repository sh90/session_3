# pip install pandas
# pip install openpyxl
import pandas as pd
import os

# Choose provider
import data_info

USE_OLLAMA = False  # Set True to use Ollama, False to use OpenAI

# Models
ollama_model = "gemma3:1b"

# Load issue tickets
df = pd.read_excel("issue_tracker.xlsx")

# Format the data into a readable prompt
ticket_prompt = "\n".join(
    f"- [Priority: {row['Priority']}] Issue Summary:{row['Summary']} (Status: {row['Status']}, Assigned to: {row['Assignee']}): {row['Description']}"
    for index, row in df.iterrows()
)

# print(ticket_prompt)
# Combine with user question
#user_question = "What are the top recurring problems and what areas need the most attention?"
user_question="How many Tickets have high priority?"
final_prompt = f"""
You are a software project analyst. Based on the issue tickets below, answer the question clearly.

Tickets:
{ticket_prompt}

Question: {user_question}

Answer:"""

print(final_prompt)
# --- Use OpenAI ---
if not USE_OLLAMA:
    import openai

    openai.api_key = data_info.open_ai_key

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.1
    )
    answer =  response.choices[0].message.content

# --- Use Ollama ---
else:
    import ollama
    response = ollama.generate(model=ollama_model, prompt=final_prompt)
    answer= response.response

# Output the result
print("\n[📊 Analysis Result]")
print(answer)
