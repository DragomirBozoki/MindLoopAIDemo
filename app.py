import streamlit as st
import pandas as pd
import time
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import snapshot_download
from llama_cpp import Llama

# ⬛ Konfiguracija stranice – MORA biti prva Streamlit komanda
st.set_page_config(page_title="MindLoop Chatbot", layout="centered")

# ⬇️ Učitavanje LLaMA modela
@st.cache_resource
def load_llama():
    return Llama(
        model_path='models/llama-2-7b-chat.Q2_K.gguf',
        n_gpu_layers=1,
        n_ctx=3900,
        n_threads=4,
        temperature=0.3,
        max_tokens=512,
        verbose=True
    )
llm = load_llama()

# ⬇️ Učitavanje FAQ baze
df = pd.read_csv("faq/ecommerce_en.csv")
questions = df["question"].tolist()
answers = df["answer"].tolist()

# ⬇️ SentenceTransformer embedder
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")
embedder = load_embedder()
corpus_embeddings = embedder.encode(questions, convert_to_tensor=True)

# ⬇️ Naslov
st.markdown("<h1 style='text-align: center; color: white;'>MindLoop Chatbot</h1>", unsafe_allow_html=True)

# ⬇️ Sidebar
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.chat = []

if "chat" not in st.session_state:
    st.session_state.chat = []

# ⬇️ Funkcija za retrieval konteksta (RAG)
def retrieve_context(user_input):
    user_emb = embedder.encode(user_input, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(user_emb, corpus_embeddings)[0]
    best_idx = int(scores.argmax())
    return answers[best_idx]

# ⬇️ Generisanje odgovora pomoću LLaMA
def generate_llama(user_input, context):
    prompt = f"""Answer the user's question **only** using the context below. Be direct and clear.

Context: "{context}"
Question: "{user_input}"
Answer:"""
    output = llm(prompt, stop=["</s>"])
    return output["choices"][0]["text"].strip()


# ⬇️ Renderovanje poruka (user/bot)
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

# ⬇️ Glavna logika četa
if user_input := st.chat_input("💬 Ask something..."):
    st.session_state.chat.append(("user", user_input))
    render_msg("user", user_input)
    with st.spinner("🤖 Thinking..."):
        context = retrieve_context(user_input)
        answer = generate_llama(user_input, context)
        placeholder = st.empty()
        shown = ""
        for c in answer:
            shown += c
            placeholder.markdown(f"""
            <div style='display: flex; justify-content: flex-end; margin: 10px 0;'>
              <div style='background-color: #3a3a3a; color: white; padding: 12px 16px; border-radius: 14px; max-width: 75%; font-family: sans-serif;'>
                {shown}
              </div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.015)
        st.session_state.chat.append(("bot", answer))

# ⬇️ Re-render prethodnih poruka
for role, msg in st.session_state.chat:
    render_msg(role, msg)

# ⬇️ Auto scroll to bottom
st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)

# ⬇️ Tamna pozadina cele stranice
st.markdown("""
<style>
    body {
        background-color: #1e1e1e;
        color: white;
    }
</style>
""", unsafe_allow_html=True)
