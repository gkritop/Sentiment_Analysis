import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from datetime import datetime


MODEL_PATH = "fine_tuned_bert"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Update predict_sentiment function to handle neutral confidence range
def predict_sentiment(text):
    if not text or not isinstance(text, str):
        return "Invalid input. Please enter a valid text."

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Confidence
    probabilities = torch.softmax(outputs.logits, dim=1)
    confidence, predicted_class = probabilities.max(dim=1)

    if confidence.item() <= 0.7:
        sentiment = "Neutral"
    else:
        sentiment = "Positive" if predicted_class.item() == 1 else "Negative"

    return sentiment, confidence.item()


# Streamlit app configuration
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="wide")

light_css = """
<style>
html, body, .main {
    background: linear-gradient(135deg, #e0f7fa, #ffffff);
    font-family: 'Segoe UI', sans-serif;
    color: #000;
}
.stTextArea textarea {
    background: rgba(255, 255, 255, 0.8);
    backdrop-filter: blur(10px);
    border: 2px solid #90caf9;
    border-radius: 12px;
    font-size: 1.1rem;
    padding: 1rem;
    color: #000;
}
.stButton button {
    background-color: #00bcd4;
    border: none;
    color: white;
    padding: 0.75rem 1.5rem;
    font-size: 1.1rem;
    border-radius: 12px;
    box-shadow: 0 0 10px #00bcd4;
    transition: 0.3s;
}
.stButton button:hover {
    background-color: #0288d1;
    box-shadow: 0 0 15px #0288d1;
}
.footer {
    margin-top: 4rem;
    text-align: center;
    font-size: 0.9rem;
    color: #555;
}
</style>
"""

st.markdown(light_css, unsafe_allow_html=True)


st.markdown("""
<h1 style='text-align: center; font-size: 3rem;
background: linear-gradient(to right, #00bcd4, #8e24aa);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
margin-bottom: 0;'>💬 Sentiment Analyzer</h1>
<p style='text-align: center; font-size: 1.2rem;'>⚡ Enter your text below to detect the vibe</p>
""", unsafe_allow_html=True)


# Input history
if 'input_history' not in st.session_state:
    st.session_state['input_history'] = []

user_input = st.text_area("Input Text", placeholder="Try something like 'I feel amazing today!'", height=150)


if st.button("🚀 Analyze Sentiment"):
    result, confidence = predict_sentiment(user_input)
    if result == "Positive":
        st.markdown(
            f"<div style='background-color:#d4edda; color:#155724; padding:10px; border-radius:5px;'>🟢 <strong>Positive</strong> – Keep smiling! (Confidence: {confidence:.2f})</div>",
            unsafe_allow_html=True
        )
        st.balloons()
    elif result == "Negative":
        st.markdown(
            f"<div style='background-color:#f8d7da; color:#721c24; padding:10px; border-radius:5px;'>🔴 <strong>Negative</strong> – That’s a rough one. (Confidence: {confidence:.2f})</div>",
            unsafe_allow_html=True
        )
    elif result == "Neutral":
        st.markdown(
            f"<div style='background-color:#fff3cd; color:#856404; padding:10px; border-radius:5px;'>⚪ <strong>Neutral</strong> – Confidence: {confidence:.2f}</div>",
            unsafe_allow_html=True
        )

    # History
    st.session_state['input_history'].append({
        'Input': user_input,
        'Sentiment': result,
        'Confidence': f"{confidence:.2f}"
    })


if st.session_state['input_history']:
    st.subheader("Input History")
    history_df = pd.DataFrame(st.session_state['input_history'])

    def highlight_sentiment(row):
        if row['Sentiment'] == 'Positive':
            return ['background-color: #d4edda; color: #155724;' if col == 'Sentiment' else '' for col in row.index]
        elif row['Sentiment'] == 'Negative':
            return ['background-color: #f8d7da; color: #721c24;' if col == 'Sentiment' else '' for col in row.index]
        elif row['Sentiment'] == 'Neutral':
            return ['background-color: #fff3cd; color: #856404;' if col == 'Sentiment' else '' for col in row.index]
        return ['' for _ in row.index]

    styled_df = history_df.style.apply(highlight_sentiment, axis=1)
    st.dataframe(styled_df, use_container_width=True)
else:
    st.subheader("Input History")
    st.write("No input history available.")


st.sidebar.title("Learn More")
st.sidebar.markdown("""
- [What is Sentiment Analysis?](https://en.wikipedia.org/wiki/Sentiment_analysis)
- [Intro to Hugging Face](https://youtu.be/QEaBAZQCtwE?si=oak7kcVvcK5uTI7l)
- [Hugging Face](https://huggingface.co/)
- [Transformers Library](https://huggingface.co/transformers/)
- [BERT Model](https://arxiv.org/abs/1810.04805)
- [IMDb Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- [Streamlit Documentation](https://docs.streamlit.io/)

""")


st.sidebar.title("Info")
st.sidebar.markdown("""
**Georgios Kritopoulos**  
- [GitHub](https://github.com/gkritop/Sentiment_Analysis#)  
- [LinkedIn](https://www.linkedin.com/in/georgios-kritopoulos)  
""")


current_date = datetime.now().strftime("%B %d, %Y")
st.markdown(f"""
<div class="footer">
    <hr style="margin-top: 3rem; margin-bottom: 1rem;">
    <p>University of Crete — Physics Department</p>
    <p>Machine Learning 2 — Project</p>
    <p>📅 {current_date}</p>
</div>
""", unsafe_allow_html=True)