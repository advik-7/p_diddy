import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.llms.base import LLM
import google.generativeai as genai
from typing import Optional, List, Mapping, Any

# Initialize the custom Gemini LLM
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

# Streamlit interface
st.title("Medical Document Retrieval and Generation with Gemini")
st.sidebar.title("Upload Document")

uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Save the uploaded file to a temporary location
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.getbuffer())

    # Load and process the document
    try:
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Split the document into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        docs = text_splitter.split_documents(documents)

        # Embed the document chunks
        st.write("### Processing Document...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(docs, embedder)

        st.success("Document successfully processed and indexed!")

        # Query input
        query = st.text_input("Enter your query:")

        if query:
            with st.spinner("Retrieving relevant information..."):
                retriever = vector_store.as_retriever()
                relevant_docs = retriever.get_relevant_documents(query)

                context = "\n".join([doc.page_content for doc in relevant_docs])

                # Generate response
                st.write("### Generating Response...")
                custom_gemini = CustomGemini(
                    temperature=0.7,
                    max_tokens=200,
                    model="gemini-model-name",
                    google_api_key=os.getenv("GEMINI_API_KEY"),
                )
                llm = CustomLLMWrapper(custom_llm=custom_gemini)
                response = llm(context + "\n\n" + query)

                st.write("### Response")
                st.success(response)
    except Exception as e:
        st.error(f"Error processing the document: {str(e)}")
    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

st.sidebar.write("---")
st.sidebar.write(
    "This application uses Google Gemini and LangChain to retrieve and generate answers from medical documents."
)
