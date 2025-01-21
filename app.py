!pip install langchain_community pypdf streamlit faiss-cpu==1.7.4 sentence-transformers google-generativeai -qq


import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from typing import Optional, List, Mapping, Any
import google.generativeai as genai
from langchain.llms.base import LLM


# Setting up the Gemini API key
os.environ["GEMINI_API_KEY"] = "AIzaSyCF2Xymk8vra8xjTh3QIIEfrLoXRIHMmLk"

# Custom class to interact with Google Gemini
class CustomGemini:
    def __init__(self, temperature: float, max_tokens: int, model: str, google_api_key: str):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        genai.configure(api_key=google_api_key)
        self.generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        self.model_instance = genai.GenerativeModel(
            model_name=model,
            generation_config=self.generation_config,
        )

    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.model_instance.generate_content(prompt)
        return response.text


# Custom LLM Wrapper for Gemini
class CustomLLMWrapper(LLM):
    custom_llm: CustomGemini

    @property
    def _llm_type(self) -> str:
        return "custom_gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self.custom_llm(prompt, stop=stop)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "temperature": self.custom_llm.temperature,
            "max_tokens": self.custom_llm.max_tokens,
            "model": self.custom_llm.model,
        }


# Process query with retrieval (RAG)
def process_query_with_retrieval(file_path: str, query: str, chat_history: Optional[List[str]] = None) -> str:
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode([t.page_content for t in texts])

    class CustomEmbeddings:
        def __init__(self, model):
            self.model = model

        def embed_documents(self, texts):
            return self.model.encode(texts)

        def embed_query(self, text):
            return self.model.encode(text)

        def __call__(self, text):
            return self.model.encode(text)

    custom_embeddings = CustomEmbeddings(model)

    db = FAISS.from_embeddings(
        text_embeddings=[(t.page_content, embedding) for t, embedding in zip(texts, embeddings)],
        embedding=custom_embeddings
    )

    custom_gemini = CustomGemini(
        temperature=0.7,
        max_tokens=512,
        model="gemini-1.5-flash",
        google_api_key=os.environ["GEMINI_API_KEY"]
    )

    wrapped_llm = CustomLLMWrapper(custom_llm=custom_gemini)

    retriever = db.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.get_relevant_documents(query)

    retrieved_content = "\n\n".join([doc.page_content for doc in retrieved_docs])

    history_content = "\n".join(chat_history) if chat_history else ""
    enhanced_query = f"""
    You are a medical diagnosis expert. When responding to queries:
    1. Provide a diagnosis or possible causes based on the details provided.
    2. If the information provided is insufficient, clearly state what additional features or details are required. Use the format: 'Question: Can you provide more details about [specific feature]?' for requesting further input.

    Here is the chat history for context:
    {history_content}

    Here are some relevant documents for context:
    {retrieved_content}

    Query: {query}
    """

    response = wrapped_llm(enhanced_query)
    return response


# Streamlit App
st.title("Medical Diagnosis Assistant with RAG")

# File uploader for PDF
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

# Input fields for query and chat history
query = st.text_input("Enter your medical query:")
chat_history_input = st.text_area("Enter previous chat history (Optional):", height=150)

# Convert chat history input to list
chat_history = chat_history_input.split("\n") if chat_history_input else []

if uploaded_file and query:
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Process the query with the uploaded file
    response = process_query_with_retrieval("temp.pdf", query, chat_history)
    st.subheader("Response:")
    st.write(response)
