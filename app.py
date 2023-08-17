import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate, LLMChain
from langchain.vectorstores import FAISS
from langchain import HuggingFaceHub
from pdfminer.high_level import extract_text
import spacy
import re

# Load the SpaCy English language model
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Remove special characters like '•'
    text = text.replace('•', '')
    
    # Remove links
    text = re.sub(r'http\S+', '', text)
    
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenize the text using SpaCy
    doc = nlp(text)
    cleaned_text = ' '.join([token.text for token in doc])
    
    return cleaned_text



def gen_embedddings(texts):
    progress_text = "Indexing documents..."
    progress_bar = st.progress(0, progress_text)
    embeddings = HuggingFaceEmbeddings()
    for i in range(len(texts)):
        FAISS.from_texts(texts[i:i+1], embeddings)
        progress_value = (i+1)/len(texts)
        if progress_value > 1.0:
            progress_value = 1.0
        progress_bar.progress(progress_value, progress_text)
    progress_bar.empty()
    st.session_state['vector_store'] = FAISS.from_texts(texts, embeddings)


def gen_response(name, question):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.1, "max_length": 150},
        huggingfacehub_api_token=token
    )

    context = ""
    for doc in st.session_state['vector_store'].similarity_search(question):
        context += doc.page_content + "\n"

    template = """
    Name of the file: {name}
    Context: {context}
    Question: {question}

    Only after understanding the question properly, answer the question based on the context above with high scrutinity.
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["name", "context", "question"]
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    ans = llm_chain.run(name=name, context=context, question=question)
    return ans


def getTexts(file):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, length_function=len
    )

    text = preprocess_text(extract_text(file))
    return splitter.split_text(text)


st.set_page_config(page_title="🤗💬 PDF Chat App",
                   initial_sidebar_state="expanded")


def clear_chat_history():
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "How may I assist you today?"
        }
    ]

with st.sidebar:
    st.title("🤗 Pdf Chat App")
    st.markdown(
        """
        ## About
        This is an LLM powered chat app that can answer questions based on the selected pdf file.
        """
    )

    def isValidToken(token):
        return token.startswith("hf_") and len(token) == 37
    if "HUGGINGFACEHUB_API_TOKEN" in st.secrets and isValidToken(st.secrets["HUGGINGFACEHUB_API_TOKEN"]):
        st.success("API token already provided!", icon="✅")
        token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    else:
        token = st.text_input(
            "Enter your Huggingface API Token", type="password")
        if not isValidToken(token):
            st.warning("Please enter a valid API token", icon="⚠️")
        else:
            st.success("API token saved!", icon="✅")

    st.button('Clear Chat History', on_click=clear_chat_history)

    st.markdown(
        """
        Made with ❤️ by [Pranav Bobde](https://github.com/Pranav-Bobde)
        """
    )


pdf = st.file_uploader("Upload a pdf file", type="pdf")

if not pdf:
    st.warning("Please upload a pdf file.", icon="⚠️")
else:
    if pdf.name[:-4] != st.session_state.get("file_name", ""):
        st.session_state["file_name"] = pdf.name[:-4]
        texts = getTexts(pdf)
        gen_embedddings(texts)  # generate embeddings for texts
        clear_chat_history()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input(disabled=not (token and pdf)):
        if prompt.strip() == "clear":
            st.session_state["messages"] = []
            st.chat_message("assistant").write("Chat cleared!")

        st.session_state["messages"].append(
            {"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        if st.session_state["messages"][-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = gen_response(pdf.name[:-4], prompt)
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response})
                    st.write(response)
