# app.py
import streamlit as st
from transformers import pipeline
import torch
import tempfile
import os

# ---- Optional integrations (LangChain, Ollama, TTS) ----
# LangChain imports (optional)
try:
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.memory import ConversationBufferMemory
    # LLM wrappers (may or may not be present depending on your langchain installation)
    try:
        from langchain.llms import HuggingFacePipeline, Ollama as LangchainOllama
    except Exception:
        # some langchain installs don't include these wrappers; we'll detect at runtime
        HuggingFacePipeline = None
        LangchainOllama = None
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False
    PromptTemplate = None
    LLMChain = None
    ConversationBufferMemory = None
    HuggingFacePipeline = None
    LangchainOllama = None

# Ollama python client (optional)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

# Optional TTS
try:
    import pyttsx3
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

# ---- Page Config & Styling ----
st.set_page_config(page_title="StoryForge AI", layout="wide")

st.markdown(
    """
    <style>
    body {
        background: linear-gradient(-45deg, #1f1c2c, #928DAB, #283c86, #45a247);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #fff;
    }
    @keyframes gradientBG {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    .title {
        font-size: 2.6em;
        font-weight: 800;
        color: #f8f9fa;
        text-align: center;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.7);
        margin-bottom: 0.2em;
    }
    .subtitle {
        font-size: 1.2em;
        text-align: center;
        color: #e0e0e0;
        margin-bottom: 2em;
    }
    .stTextArea textarea {
        background-color: #2b2b2b !important;
        color: #ffffff !important;
        border-radius: 10px;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: #2b2b2b !important;
        border-radius: 8px;
    }
    .stButton button {
        background: linear-gradient(45deg, #6a11cb, #2575fc);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        transition: 0.3s;
    }
    .stButton button:hover {
        transform: scale(1.05);
        background: linear-gradient(45deg, #2575fc, #6a11cb);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">✍️ StoryForge AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Your Creative Co-Writer powered by Local & Open-Source LLMs</div>', unsafe_allow_html=True)

# ---- Sidebar / Controls ----
st.sidebar.header("⚙️ Model & Generation Settings")
model_choice = st.sidebar.selectbox(
    "Choose backend",
    options=(["gpt2", "distilgpt2"] +
             (["ollama/llama2", "ollama/mistral"] if OLLAMA_AVAILABLE else [])),
    help="Pick Hugging Face (gpt2/distilgpt2) or Ollama models if installed."
)

max_new_tokens = st.sidebar.slider("Max new tokens", 50, 800, 180)
temperature = st.sidebar.slider("Creativity (temperature)", 0.1, 1.2, 0.9)
top_p = st.sidebar.slider("Top-p (nucleus)", 0.1, 1.0, 0.95)
num_return_sequences = st.sidebar.selectbox("Responses to generate", [1, 2, 3])

# ---- Hugging Face pipeline loader (cached) ----
@st.cache_resource
def load_hf_generator(model_name_str):
    device = 0 if torch.cuda.is_available() else -1
    # We will create a simple generator pipeline; we set no fixed max_length here,
    # generation length will be handled when we call the pipeline.
    generator = pipeline(
        "text-generation",
        model=model_name_str,
        tokenizer=model_name_str,
        device=device,
        trust_remote_code=False
    )
    return generator

# ---- LangChain prompt template & memory ----
if LANGCHAIN_AVAILABLE:
    story_prompt = PromptTemplate(
        input_variables=["starter_text", "genre", "style"],
        template=(
            "You are a creative writer. Continue the story below in the {genre} genre, "
            "using a {style} tone. Keep continuity and characters consistent. "
            "Add vivid descriptions.\n\nSTARTER:\n{starter_text}\n\nCONTINUATION:"
        )
    )
    # IMPORTANT: ConversationBufferMemory must know which single input key to record.
    # We set input_key="starter_text" so LangChain can find what to save/recall.
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="starter_text")
else:
    # fallback prompt builder (same text, manual)
    def make_prompt(starter_text, genre, style):
        return (
            f"You are a creative writer. Continue the story below in the {genre} genre, "
            f"using a {style} tone. Keep continuity and characters consistent. "
            f"Add vivid descriptions.\n\nSTARTER:\n{starter_text}\n\nCONTINUATION:"
        )

# ---- UI: main area ----
st.subheader("📝 Write a starting passage:")
starter = st.text_area("Your passage:", height=180, value="Once upon a time in a city of glass,")
col1, col2 = st.columns([2,1])

with col2:
    genre = st.selectbox("Genre", ["Fantasy","Horror","Sci-Fi","Romance","Thriller","Mystery","Comedy","Slice of Life"])
    style = st.selectbox("Style / Tone", ["Dramatic","Playful","Poetic","Dark","Casual","Shakespearean","Modern","Minimalist"])

gen_btn = st.button("✨ Generate continuation")

# ---- Helper: generate using Ollama direct client (fallback) ----
def generate_with_ollama_direct(model_name, prompt_text, temperature, top_p):
    results = []
    # Ollama python client returns nested structure; call repeatedly if multiple responses requested
    for _ in range(num_return_sequences):
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt_text}],
            options={"temperature": float(temperature), "top_p": float(top_p)}
        )
        # Ollama response: {"message": {"content": "..."}}
        content = response.get("message", {}).get("content", "")
        results.append(content)
    return results

# ---- Helper: generate using Hugging Face pipeline direct (fallback) ----
def generate_with_hf_direct(generator, prompt_text, max_new_tokens, temperature, top_p, num_return_sequences):
    outputs = generator(
        prompt_text,
        max_length = len(prompt_text.split()) + int(max_new_tokens),
        do_sample = True,
        temperature = float(temperature),
        top_p = float(top_p),
        num_return_sequences = int(num_return_sequences),
        return_full_text = True
    )
    results = []
    for out in outputs:
        full_text = out.get("generated_text", "")
        continuation = full_text[len(prompt_text):].strip() if full_text.startswith(prompt_text) else full_text.strip()
        results.append(continuation)
    return results

# ---- Main generation logic ----
if gen_btn and starter.strip():
    results = []

    if LANGCHAIN_AVAILABLE:
        # Build an LLM object for LangChain. Prefer langchain's wrappers if available,
        # otherwise create a small wrapper-like object using the HF pipeline and a very small shim.
        if model_choice.startswith("ollama/") and OLLAMA_AVAILABLE:
            model_name = model_choice.replace("ollama/", "")
            # If LangChain's Ollama wrapper is present, use it; otherwise we'll call ollama directly via a small shim.
            if LangchainOllama is not None:
                llm = LangchainOllama(model=model_name, temperature=temperature)
                chain = LLMChain(llm=llm, prompt=story_prompt, memory=memory)
                with st.spinner(f"Generating with LangChain + Ollama ({model_name})..."):
                    # use predict so we can supply multiple named args
                    for _ in range(num_return_sequences):
                        continuation = chain.predict(starter_text=starter, genre=genre, style=style)
                        results.append(continuation)
            else:
                # LangChain Ollama wrapper not installed -> fallback to direct ollama python client
                prompt_text = story_prompt.format(starter_text=starter, genre=genre, style=style)
                with st.spinner(f"Generating with Ollama (direct client) {model_name}..."):
                    results = generate_with_ollama_direct(model_name, prompt_text, temperature, top_p)
                    # If memory is available, manually append to memory buffer so subsequent LangChain chains (if any) can read it.
                    try:
                        # memory expects a dict with inputs and outputs keyed by the prompt input key
                        memory.save_context({"starter_text": starter}, {"output": "\n\n".join(results)})
                    except Exception:
                        pass
        else:
            # Hugging Face backend through LangChain if possible
            # 1) load HF pipeline
            generator = load_hf_generator(model_choice)
            if HuggingFacePipeline is not None:
                # Wrap the transformer pipeline as a LangChain LLM
                hf_llm = HuggingFacePipeline(pipeline=generator)
                chain = LLMChain(llm=hf_llm, prompt=story_prompt, memory=memory)
                with st.spinner("Generating with LangChain + Hugging Face..."):
                    for _ in range(num_return_sequences):
                        continuation = chain.predict(starter_text=starter, genre=genre, style=style)
                        results.append(continuation)
            else:
                # HuggingFacePipeline wrapper missing; fallback to direct pipeline calls
                prompt_text = story_prompt.format(starter_text=starter, genre=genre, style=style)
                with st.spinner("Generating with Hugging Face (direct pipeline)..."):
                    results = generate_with_hf_direct(generator, prompt_text, max_new_tokens, temperature, top_p, num_return_sequences)
                    try:
                        memory.save_context({"starter_text": starter}, {"output": "\n\n".join(results)})
                    except Exception:
                        pass
    else:
        # LangChain not installed: fallback to earlier direct approach (Hugging Face or Ollama)
        prompt_text = make_prompt(starter, genre, style) if not LANGCHAIN_AVAILABLE else story_prompt.format(starter_text=starter, genre=genre, style=style)

        if model_choice.startswith("ollama/") and OLLAMA_AVAILABLE:
            model_name = model_choice.replace("ollama/", "")
            with st.spinner(f"Generating with Ollama (direct client) {model_name}..."):
                results = generate_with_ollama_direct(model_name, prompt_text, temperature, top_p)
        else:
            with st.spinner("Generating with Hugging Face (direct pipeline)..."):
                generator = load_hf_generator(model_choice)
                results = generate_with_hf_direct(generator, prompt_text, max_new_tokens, temperature, top_p, num_return_sequences)

    # ---- Show outputs ----
    for i, continuation in enumerate(results):
        st.markdown(f"### 🌟 Result #{i+1}")
        st.success(continuation)
        st.download_button(f"💾 Download story #{i+1}", continuation, file_name=f"story_{i+1}.txt")

        if TTS_AVAILABLE:
            st.write("🔊 Listen:")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp.close()
            engine = pyttsx3.init()
            engine.save_to_file(continuation, tmp.name)
            engine.runAndWait()
            st.audio(tmp.name)
            try:
                os.remove(tmp.name)
            except Exception:
                pass
