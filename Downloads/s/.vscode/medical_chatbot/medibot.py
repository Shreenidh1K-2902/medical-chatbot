import os
import streamlit as st

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

# ------------------------- Page Configuration ------------------------- #
st.set_page_config(
    page_title="MediBot - Your Health Assistant",
    page_icon="🧠",
    layout="centered",
)

# ------------------------- Custom CSS ------------------------- #
st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f0f2f6;
    }
    .chat-bubble {
        background-color: #e6f4f1;
        padding: 10px 15px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 85%;
    }
    .user-bubble {
        background-color: #cce5ff;
        align-self: flex-end;
    }
    .bot-bubble {
        background-color: #f7f7f9;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.9em;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        huggingfacehub_api_token=HF_TOKEN,
        model_kwargs={"max_length": 512}
    )

def format_response(result, source_documents):
    if any(line.strip().startswith(("-", "*", "1.")) for line in result.splitlines()):
        formatted_result = "\n".join(
            f"- {line.strip()}" if not line.strip().startswith("-") else line
            for line in result.strip().splitlines()
        )
    else:
        formatted_result = result.strip()

    if source_documents:
        formatted_sources = "\n\n**Source Documents:**\n" + "\n".join(
            f"- **{doc.metadata.get('source', 'Unknown Source')}**:\n  {doc.page_content[:300].strip()}..."
            for doc in source_documents
        )
    else:
        formatted_sources = "\n\n_No source documents returned._"

    return f"{formatted_result}\n{formatted_sources}"

def main():
    st.markdown("<h1 style='text-align: center; color: #2c3e50;'>🤖 MediBot</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 18px;'>Your intelligent assistant for contextual Q&A</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=100)
        st.markdown("### 🤝 How can MediBot help?")
        st.markdown("- Medical Q&A\n- Context-based replies\n- No hallucination\n\n_Only answers from your document context._")
        st.markdown("---")
        st.markdown("© 2025 MediBot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Show chat history
    for msg in st.session_state.messages:
        role_class = "user-bubble" if msg['role'] == 'user' else "bot-bubble"
        st.markdown(f"<div class='chat-bubble {role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

    # Get prompt input from the user
    prompt = st.chat_input("Ask a medical question...")

    if prompt:
        # Append the user's message to session state
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
        Don't provide anything outside the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            # Create the QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(HUGGINGFACE_REPO_ID, HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            # Get response from the chain
            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]
            formatted = format_response(result, source_documents)

            # Append the assistant's response to session state
            st.session_state.messages.append({'role': 'assistant', 'content': formatted})

            # Display the assistant's response
            st.markdown(f"<div class='chat-bubble bot-bubble'>{formatted}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Footer with information about the bot
    st.markdown("<div class='footer'>Built with ❤️ using LangChain + Streamlit</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
