import streamlit as st
from transformers import BartTokenizer, BartForConditionalGeneration

# Initialize the tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Define the Streamlit app interface
st.title("AI-Based Text Summarizer")
st.subheader("Enter a prompt, and the model will generate a summary.")

# User input for the text prompt
prompt = st.text_area("Input Prompt")

if st.button("Generate Summary"):
    if prompt:
        # Tokenize the input text
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)

        # Generate the summary
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=4,
            max_length=200,
            early_stopping=True
        )

        # Decode and display the summary
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.subheader("Generated Summary:")
        st.write(summary)
    else:
        st.warning("Please enter a valid prompt for summarization.")

