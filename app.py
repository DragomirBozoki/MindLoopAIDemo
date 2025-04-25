import streamlit as st
import pandas as pd
import time
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import snapshot_download
from llama_cpp import Llama
import os

# ⬛ Konfiguracija stranice
st.set_page_config(page_title="MindLoop Chatbot", layout="centered")

@st.cache_resource
def load_llama():
    model_dir = snapshot_download(repo_id="dragomir01/chatbotweb")
    model_path = os.path.join(model_dir, "llama-2-7b-chat.Q2_K.gguf")
    return Llama(
        model_path=model_path,
        n_gpu_layers=1,
        n_ctx=3900,
        n_threads=4,
        temperature=0.25,
        max_tokens=2048,
        verbose=False
    )

llm = load_llama()

# ⬇️ Učitavanje FAQ baze
df = pd.read_csv("faq/ecommerce_en.csv")
questions = df["question"].tolist()
answers = df["answer"].tolist()

# ⬇️ Učitavanje embeddera
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")
embedder = load_embedder()
corpus_embeddings = embedder.encode(questions, convert_to_tensor=True)

# ⬇️ Naslov
st.markdown("<h1 style='text-align: center; color: white;'>MindLoop Chatbot</h1>", unsafe_allow_html=True)

# ⬇️ Sidebar
with st.sidebar:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat = []
    
    st.markdown("---")
    st.markdown("🧠 **Model Info**")
    st.markdown("""
    This chatbot uses a **Retrieval-Augmented Generation (RAG)** approach with:
    
    - 🦙 **LLaMA 2** local model for answer generation  
    - 🔎 **SentenceTransformer** for semantic search  
    - 📦 Custom **FAQ database** focused on **e-commerce**
    
    If a relevant context is found, the model uses it. Otherwise, it falls back to LLaMA's own knowledge.
    """)

# ⬇️ Inicijalizacija sesije
if "chat" not in st.session_state:
    st.session_state.chat = []

# ⬇️ RAG funkcija sa scoringom
def retrieve_context(user_input):
    user_emb = embedder.encode(user_input, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(user_emb, corpus_embeddings)[0]
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])
    return answers[best_idx], best_score

# ⬇️ LLaMA generacija
def generate_llama(prompt):
    output = llm(prompt, stop=["</s>"])
    return output["choices"][0]["text"].strip()

# ⬇️ Render poruka
def render_msg(role, msg):
    align = "flex-start" if role == "user" else "flex-end"
    bg = "#2a2a2a" if role == "user" else "#3a3a3a"
    text_color = "white"
    st.markdown(f"""
    <div style='display: flex; justify-content: {align}; margin: 10px 0;'>
      <div style='background-color: {bg}; color: {text_color}; padding: 12px 16px; border-radius: 14px; max-width: 75%; font-family: sans-serif;'>
        {msg}
      </div>
    </div>
    """, unsafe_allow_html=True)

# ⬇️ Render svih prethodnih poruka
for role, msg in st.session_state.chat:
    render_msg(role, msg)

# ⬇️ Chat logika
if (user_input := st.chat_input("💬 Ask something...")):
    st.session_state.chat.append(("user", user_input))
    with st.spinner("🤖 Thinking..."):
        context, score = retrieve_context(user_input)

        if score < 0.6:
            # Fallback na LLM znanje
            prompt = f"You are a helpful assistant. Answer the following question using your own knowledge.\n\nQuestion: {user_input}\nAnswer:"
        else:
            # RAG prompt
            prompt = f"""You are a helpful assistant. Use ONLY the context to answer the question. Be short and clear.

Context: "{context}"
Question: "{user_input}"
Answer:"""
        
        answer = generate_llama(prompt)
        st.session_state.chat.append(("bot", answer))

# ⬇️ Auto scroll
st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

# ⬇️ Tamna tema
st.markdown("""
<style>
    body {
        background-color: #1e1e1e;
        color: white;
    }
</style>
""", unsafe_allow_html=True)
