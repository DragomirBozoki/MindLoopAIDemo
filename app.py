import streamlit as st
import pandas as pd
import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

# --------------- Load Models ----------------
@st.cache_resource
def load_models():
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-small")
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    retriever = SentenceTransformer("all-MiniLM-L6-v2")
    return tokenizer, model, retriever

tokenizer, model, retriever = load_models()

# --------------- Load FAQ ----------------
@st.cache_data
def load_faq():
    try:
        df = pd.read_csv("faq/faq_ecommerce/faq_ecommerce_en.csv")
        return df.dropna().reset_index(drop=True)
    except FileNotFoundError:
        st.error("❌ FAQ file not found! Make sure `faq/faq_ecommerce/faq_ecommerce_en.csv` exists.")
        return pd.DataFrame(columns=["question", "answer"])

faq_df = load_faq()

# --------------- Setup ----------------
st.set_page_config(page_title="MindLoop AI", layout="centered")
st.title("🧠 MindLoop AI Chat")
st.write("Ask something related to **E-commerce FAQ** (EN only)")

# --------------- Session State ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --------------- Chat Interface ----------------
def generate_answer(user_input):
    if faq_df.empty:
        return "FAQ database is not loaded."

    # Embed user input and all questions
    user_embedding = retriever.encode(user_input, convert_to_tensor=True)
    faq_embeddings = retriever.encode(faq_df["question"].tolist(), convert_to_tensor=True)

    # Find best match
    cos_scores = util.pytorch_cos_sim(user_embedding, faq_embeddings)[0]
    top_idx = torch.argmax(cos_scores).item()

    context = faq_df.iloc[top_idx]["answer"]
    prompt = f"question: {user_input} context: {context}"

    # Generate answer
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    output = model.generate(**inputs, max_length=100)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer

# --------------- User Input ----------------
user_input = st.text_input("You:", key="input_field", placeholder="Type your question here...")

if user_input:
    # Show chat history
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.chat_message("user", avatar="🧑").write(q)
        st.chat_message("ai", avatar="🤖").write(a)

    # User message
    st.chat_message("user", avatar="🧑").write(user_input)

    # Thinking bar appears *after* input
    with st.spinner("🤖 Thinking..."):
        answer = generate_answer(user_input)

    # Bot response
    st.chat_message("ai", avatar="🤖").write(answer)
    st.session_state.chat_history.append((user_input, answer))
