import os
import platform
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import fitz  # PyMuPDF

# Try importing chromadb if not on Streamlit Cloud
is_streamlit_cloud = platform.system() == "Linux"
if not is_streamlit_cloud:
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(
    page_title="AeroMate - Your Aerospace Assistant",
    page_icon="🚀",
    layout="centered"
)

# Configure Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('models/gemini-1.5-flash-latest')

# Helper
def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

# Initialize sessions
if "recommender_session" not in st.session_state:
    st.session_state.recommender_session = model.start_chat(history=[])

if "chatbot_session" not in st.session_state:
    st.session_state.chatbot_session = model.start_chat(history=[])

if "rag_chunks" not in st.session_state:
    st.session_state.rag_chunks = []
    st.session_state.rag_sources = []
    st.session_state.rag_collection = None
    st.session_state.uploaded_pdfs_memory = {}
    st.session_state.rag_db_built = False

# Tab selection
active_tab = st.radio(
    "Choose a mode",
    ["🔍 Material Recommender", "🤖 Chatbot", "📄 Research Paper Scraper", "📄 Upload & Summarize PDF", "💬 Chat with My Papers"],
    horizontal=True
)

# ---------------------- Tab 1 ----------------------
if active_tab == "🔍 Material Recommender":
    st.title("🔍 Aerospace Material Recommender")

    component = st.selectbox("Select the rocket component:", [
        "Fuselage", "Nozzle", "Fuel Tank", "Engine", "Wings", "Payload Structure"
    ])

    requirement = st.selectbox("Select the key requirement:", [
        "High Temperature Resistance", "Lightweight", "High Strength",
        "Corrosion Resistance", "Low Cost", "Reusability"
    ])

    if st.button("Suggest Material"):
        prompt = f"Suggest suitable materials for a rocket {component.lower()} that needs {requirement.lower()}. Mention 2–3 material options and justify briefly."
        st.chat_message("user").markdown(prompt)
        response = st.session_state.recommender_session.send_message(prompt)
        with st.chat_message("assistant"):
            st.markdown(response.text)

    for msg in st.session_state.recommender_session.history:
        with st.chat_message(translate_role_for_streamlit(msg.role)):
            st.markdown(msg.parts[0].text)

# ---------------------- Tab 2 ----------------------
elif active_tab == "🤖 Chatbot":
    st.title("🤖 Chat with AeroMate")

    for msg in st.session_state.chatbot_session.history:
        with st.chat_message(translate_role_for_streamlit(msg.role)):
            st.markdown(msg.parts[0].text)

    user_prompt = st.chat_input("Ask AeroMate anything about aerospace...")
    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        response = st.session_state.chatbot_session.send_message(user_prompt)
        with st.chat_message("assistant"):
            st.markdown(response.text)

# ---------------------- Tab 3 ----------------------
elif active_tab == "📄 Research Paper Scraper":
    st.title("📄 Research Paper Scraper & Summarizer (ScienceDirect)")

    url = st.text_input("Paste ScienceDirect journal issue URL:")

    if url:
        try:
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            soup = BeautifulSoup(res.text, "html.parser")
            articles = []
            cards = soup.select("li.js-article-list-item")

            for card in cards:
                title_tag = card.select_one("h2 a")
                if not title_tag:
                    continue
                title = title_tag.text.strip()
                link = urljoin("https://www.sciencedirect.com", title_tag.get("href"))
                open_access = bool(card.select_one(".access-indicator.OpenAccess"))
                articles.append({"title": title, "link": link, "open_access": open_access})

            if not articles:
                st.warning("No articles found.")
            else:
                selected_title = st.selectbox("Choose an article:", [a["title"] for a in articles])
                selected_article = next(a for a in articles if a["title"] == selected_title)

                st.markdown(f"🔗 [Open in browser]({selected_article['link']})")
                if selected_article["open_access"]:
                    st.success("✅ Open Access")
                else:
                    st.info("ℹ️ Not Open Access")

                if st.button("Summarize"):
                    try:
                        article_res = requests.get(selected_article["link"], headers={"User-Agent": "Mozilla/5.0"})
                        article_soup = BeautifulSoup(article_res.text, "html.parser")
                        if selected_article["open_access"]:
                            paragraphs = article_soup.select("div.abstract.author p") + article_soup.select("section > p")
                            full_text = "\n".join(p.text.strip() for p in paragraphs if p.text.strip())
                        else:
                            abstract_tag = article_soup.select_one("div.abstract.author")
                            full_text = abstract_tag.text.strip() if abstract_tag else "No abstract available."

                        if not full_text:
                            st.warning("Couldn't extract text.")
                        else:
                            response = model.generate_content(f"Summarize this article:\n\n{full_text[:8000]}")
                            st.subheader("🧠 Gemini Summary")
                            st.markdown(response.text)

                    except Exception as e:
                        st.error(f"Error summarizing article: {e}")

        except Exception as e:
            st.error(f"Error accessing page: {e}")

# ---------------------- Tab 4 ----------------------
elif active_tab == "📄 Upload & Summarize PDF":
    st.title("📄 Upload Research Paper PDF & Summarize")
    uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

    if uploaded_file:
        with st.spinner("Reading PDF..."):
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = "".join(page.get_text() for page in doc)

        if not text.strip():
            st.warning("No text extracted.")
        else:
            response = model.generate_content(f"Summarize this research paper:\n\n{text[:8000]}")
            st.subheader("🧠 Gemini Summary")
            st.markdown(response.text)

# ---------------------- Tab 5 ----------------------
elif active_tab == "💬 Chat with My Papers":
    st.title("💬 Chat with My Papers (RAG with ChromaDB)")

    if is_streamlit_cloud:
        st.warning("⚠️ This feature is disabled on Streamlit Cloud. Please run this app locally to use ChromaDB-based RAG.")
    else:
        uploaded_pdfs = st.file_uploader("Upload multiple PDFs", type=["pdf"], accept_multiple_files=True)

        if uploaded_pdfs:
            with st.spinner("Reading and chunking PDFs..."):
                for file in uploaded_pdfs:
                    if file.name not in st.session_state.uploaded_pdfs_memory:
                        doc = fitz.open(stream=file.read(), filetype="pdf")
                        raw_text = "".join(page.get_text() for page in doc)
                        chunks = [raw_text[i:i+1000] for i in range(0, len(raw_text), 1000)]
                        st.session_state.uploaded_pdfs_memory[file.name] = chunks

                st.session_state.rag_chunks = []
                st.session_state.rag_sources = []
                for fname, chunks in st.session_state.uploaded_pdfs_memory.items():
                    st.session_state.rag_chunks.extend(chunks)
                    st.session_state.rag_sources.extend([fname] * len(chunks))

        if st.session_state.rag_chunks and not st.session_state.rag_db_built:
            with st.spinner("Embedding and storing in ChromaDB..."):
                chroma_client = chromadb.Client()
                embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
                collection = chroma_client.get_or_create_collection(name="my-papers", embedding_function=embed_fn)

                for i, chunk in enumerate(st.session_state.rag_chunks):
                    collection.add(
                        documents=[chunk],
                        ids=[f"chunk-{i}"],
                        metadatas=[{"source": st.session_state.rag_sources[i]}]
                    )

                st.session_state.rag_collection = collection
                st.session_state.rag_db_built = True

            st.success(f"✅ Stored {len(st.session_state.rag_chunks)} chunks from {len(st.session_state.uploaded_pdfs_memory)} files.")

        if st.session_state.rag_collection:
            user_question = st.text_input("Ask a question about your uploaded papers:")
            if user_question:
                results = st.session_state.rag_collection.query(query_texts=[user_question], n_results=5)
                if results['documents']:
                    retrieved_text = "\n\n".join(results['documents'][0])
                    prompt = f"""Use the following context from uploaded PDFs to answer the user's question.

Context:
{retrieved_text}

Question:
{user_question}

Answer concisely, citing content when possible."""
                    with st.spinner("Thinking with Gemini..."):
                        response = model.generate_content(prompt)
                        st.subheader("🧠 Gemini's Answer")
                        st.markdown(response.text)
