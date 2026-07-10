from openai import OpenAI

model = "gpt-4o-mini"

client = OpenAI()

def get_response(conversation):
    return (
        client.chat.completions.create(
            model=model,
            messages=conversation,
            temperature=0,
            max_completion_tokens=100
        ).choices[0].message.content
    )

sys_msg = """
    You are a chatbot that answers questions asked by tourists.
    If a question that is not related to tourism is asked, respond with 'Sorry, I can only answer questions on tourism.'
"""

user_questions = []
question = ""

# The user can ask multiple questions
while question != "q":
    question = input("Enter a tourism question to ask (q to quit): ")
    if (question != "q"):
        user_questions.append(question)

conversation = [
    {"role": "system", "content": sys_msg},
]

# Add the question to the conversation, and get a response from the AI that is also added to the conversation
for q in user_questions:
    print("Question: " + q)
    user_dict = {"role": "user", "content": q}
    conversation.append(user_dict)

    response = get_response(conversation)

    assistant_dict = {"role": "assistant", "content": response}
    print("Response: " + response)
    print()
    conversation.append(assistant_dict)

