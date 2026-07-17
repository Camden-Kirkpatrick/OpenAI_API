"""
Ask the model a series of questions in one ongoing conversation.
Each question and its answer are appended to 'messages', so the full
history is present in every call and the model retains context across turns
"""

from openai import OpenAI

client = OpenAI()

messages = [
    {"role": "system", "content": "You are a helpful math tutor that speaks concisely."}
]
user_msgs = ["Explain what pi is.", "Explain what the golden ratio is."]

# Loop over the user questions
for q in user_msgs:
    print("User: ", q)
    
    # Create a dictionary for the user message from q and append to messages
    user_dict = {"role": "user", "content": q}
    messages.append(user_dict)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        max_completion_tokens=100
    )
    
    # Append the assistant's message to messages
    assistant_dict = {"role": "assistant", "content": response.choices[0].message.content}
    messages.append(assistant_dict)
    print("Assistant: ", response.choices[0].message.content, "\n")