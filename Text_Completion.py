import streamlit as st
import torch
from transformers import AutoTokenizer
import pickle

# Load the saved model and tokenizer
def load_model_and_tokenizer(pkl_file):
    with open(pkl_file, "rb") as f:
        model = pickle.load(f)
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")  # Adjust for your model
    return model, tokenizer

# Generate chatbot response
def generate_response(model, tokenizer, user_input):
    # Tokenize the input
    inputs = tokenizer(user_input, return_tensors="pt")
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Generate top prediction (adjust based on your model's task, e.g., masked language filling)
    logits = outputs.logits
    predicted_token_id = torch.argmax(logits[0], dim=-1)
    response = tokenizer.decode(predicted_token_id, skip_special_tokens=True)
    return response

# Streamlit UI
def main():
    st.title("Chatbot with Hugging Face and Streamlit")
    st.write("This chatbot uses a pre-trained Hugging Face model saved in `.pkl` format.")

    # Load the model and tokenizer
    model_file = "/Users/sudarshanc/Downloads/trained_model.pkl"  # Replace with your .pkl file path
    st.sidebar.title("Model Configuration")
    st.sidebar.write(f"Using model file: `{model_file}`")
    model, tokenizer = load_model_and_tokenizer(model_file)

    # Chat interface
    user_input = st.text_input("You:", "")
    if st.button("Send"):
        if user_input.strip():
            response = generate_response(model, tokenizer, user_input)
            st.text_area("Chatbot:", value=response, height=100, max_chars=None)

if __name__ == "__main__":
    main()
