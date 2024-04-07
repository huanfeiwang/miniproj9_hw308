"""
import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Title and introduction
st.title("Sentiment Analysis Tool")
st.write("This tool uses a machine learning model to analyze the sentiment of the input text.")

# Tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# UI with sidebar for input
st.sidebar.title("User Input")
input_text = st.sidebar.text_area("Enter the text you want to analyze:", height=6)

# Analyze sentiment button
generate_button = st.sidebar.button("Evaluate Sentiment")

if generate_button and input_text:
    st.sidebar.write("Analyzing sentiment...")
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        sentiment_score, predicted_class_id = torch.max(probabilities, dim=1)

    sentiment = model.config.id2label[predicted_class_id.item()]

    # Debugging: Output raw scores and probabilities
    st.write("Logits:", logits)
    st.write("Probabilities:", probabilities)

    st.subheader("Analysis Result")
    st.write("Sentiment:", sentiment)
    st.write(f"Confidence Score: {sentiment_score.item():.4f}")
    st.progress(sentiment_score.item())



"""
import streamlit as st
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

# Title and introduction
st.title("Sentiment Analysis Tool")
st.write("This tool uses a machine learning model to analyze the sentiment of the input text.")

# Tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# UI with sidebar for input
st.sidebar.title("User Input")
input_text = st.sidebar.text_area("Enter the text you want to analyze:", height=6)

# Analyze sentiment button
generate_button = st.sidebar.button("Evaluate Sentiment")

# Processing the input text
if generate_button and input_text:
    st.sidebar.write("Analyzing sentiment...")
    # Tokenize the input text and prepare it as input to the model
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        sentiment_score, predicted_class_id = torch.max(probabilities, dim=1)

    # Mapping model output labels to human-readable sentiments
    label_map = {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive'
    }

    sentiment = label_map[model.config.id2label[predicted_class_id.item()]]

    st.subheader("Analysis Result")
    st.write("Sentiment:", sentiment.capitalize())
    st.write(f"Confidence Score: {sentiment_score.item():.4f}")
    st.progress(sentiment_score.item())

    if sentiment == "positive":
        st.success("The text is positive.")
    elif sentiment == "negative":
        st.error("The text is negative.")
    else:
        st.warning("The text has a neutral sentiment.")
