import streamlit as st
from openai import OpenAI

# Access the OpenAI API key from the secrets
api_key = st.secrets["openai_api_key"]

# Set up the OpenAI API client
client = OpenAI(api_key=api_key)

def generate_completion(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": "You are an helpful assistant\n"
                }
            ]
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": prompt
                }
            ]
            },
        ],
        max_tokens=100
    )
    return response.choices[0].message.content

# A simple Streamlit UI
st.title("OpenAI GPT-3.5 Turbo Demo")
prompt = st.text_input("Enter a prompt:")

if st.button("Generate"):
    if prompt:
        completion = generate_completion(prompt)
        st.markdown(completion)
    else:
        st.write("Please enter a prompt.")