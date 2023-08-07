# simplePDF - A FREE PDF Chat App powered by LLMs

This is a PDF chat app which you can run on your local machine. The app allows users to upload any PDF and ask questions. The app is built using Langchain, Streamlit and Hugging Face Q&A LLM.

## Features
- Upload any PDF file
- Ask questions about the PDF

## How to Use
1. Clone the repository:
```
git clone https://github.com/sudan94/chat-pdf-hugginface.git
```
2. Install the required packages:
```
pip install streamlit langchain huggingface-hub pypdf2 sentence-transformers faiss-cpu
```
3. Add the huggingface token:
    1. rename .streamlit/secrets.local.toml to .streamlit/secrets.toml
    2. paste the token inside the double quotes
    #NOTE: The quotes are necessary

4. Run the app:
```
streamlit run app.py
```
5. Upload a PDF file and ask questions about it.
