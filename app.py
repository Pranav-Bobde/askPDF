import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import PromptTemplate, LLMChain
from langchain.vectorstores import FAISS
from langchain import HuggingFaceHub
from PyPDF2 import PdfReader


def get_answer(name, texts, question):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.1, "max_length": 64},
    )  # type: ignore

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_texts(texts, embeddings)
    context = db.similarity_search(question)

    template = """
    You are the pdf named "{name}". Your task is to extract information and provide 
    accurate answers from the PDF documents uploaded by users. You will be presented 
    with context and question prompts related to the documents. Your goal is to 
    understand the context, extract relevant information, and generate correct 
    and informative responses. Remember to consider the context and provide 
    answers that are coherent, relevant, and knowledgeable.
    Following are the context and question prompts:

    Context: {context}

    Question: `{question}`

    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["name", "context", "question"]
    )

    llm_chain = LLMChain(prompt=prompt, llm=llm)
    ans = llm_chain.run(name=name, context=context, question=question)
    return ans


def getTexts(file):
    text = ""
    pdf_reader = PdfReader(file)

    for page in pdf_reader.pages:
        text += page.extract_text()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100, length_function=len
    )

    return splitter.split_text(text)

st.set_page_config(page_title="🤗💬 PDF Chat App", initial_sidebar_state="expanded")


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

    def clear_chat_history():
        st.session_state["messages"] = [{"role": "assistant", "content": "How may I assist you today?"}]
    st.button('Clear Chat History', on_click=clear_chat_history)

    st.markdown(
        """
        Made with ❤️ by [Pranav Bobde](https://github.com/Pranav-Bobde)
        """
    )


pdf = st.file_uploader("Upload a pdf file", type="pdf")

if pdf:
    texts = getTexts(pdf)

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

    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    if st.session_state["messages"][-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_answer(pdf.name[:-4], texts, prompt)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": response})
                st.write(response)