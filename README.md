Here’s a properly structured `README.md` file for your Streamlit-based RAG (Retrieval-Augmented Generation) PDF QA project using LangChain, HuggingFace embeddings, FAISS, and Groq's Gemma model.

---

# 🧠 PDF Question Answering using LangChain, HuggingFace, FAISS & Groq

This project demonstrates how to build a **PDF Question-Answering** application using the following tools:

* 📄 **PyPDF2** for PDF parsing
* 🧩 **LangChain** for RAG pipeline
* 🤗 **HuggingFace Embeddings**
* 🔍 **FAISS** for vector storage and retrieval
* 💬 **Groq API (Gemma 2 9B model)** for generating answers
* 🌐 **Streamlit** for the interactive user interface

---

## 🚀 Features

* Extract text from a PDF file.
* Chunk and embed the document using HuggingFace's Sentence Transformer.
* Store and query embeddings using FAISS.
* Ask natural language questions about the PDF.
* Use a Groq-hosted LLM (Gemma 2 9B IT) for context-aware answers.

---

## 📁 Project Structure

```
├── app.py               # Main Streamlit application
├── requirements.txt     # Dependencies
├── README.md            # This documentation
```

---

## 📦 Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/pdf-qa-groq.git
cd pdf-qa-groq
```

2. **Create a virtual environment and activate it**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

Add this to your `requirements.txt`:

```txt
streamlit
PyPDF2
langchain
langchain-community
sentence-transformers
faiss-cpu
```

---

## 🔑 Setup API Key

Make sure to store your Groq API key securely. You can either:

* Set it as an environment variable:

```bash
export GROQ_API_KEY=your_groq_api_key_here
```

* Or hardcode (for development only) in the script:

```python
groqapi = 'your_groq_api_key_here'
```

---

## 🛠️ How It Works

1. **Upload PDF**: Parses the PDF using `PyPDF2`.
2. **Split Text**: Splits the text using `RecursiveCharacterTextSplitter`.
3. **Embed & Store**: Embeds the chunks with HuggingFace and stores them in a FAISS vector store.
4. **Query**: User inputs a question → relevant chunks are retrieved → sent to Groq's LLM for answer generation.

---

## 💻 Usage

1. Replace the `uploaded_file` path in `app.py` with your local PDF path or modify the code to allow file upload via Streamlit.

2. Run the app:

```bash
streamlit run app.py
```

3. Ask questions in the text input and get answers based on the PDF content.

---

## 🧠 Example Prompt Template

```txt
You are a helpful assistant. Answer the question using only the context below.
If the answer is not present, just say no. Do not try to make up an answer.

Context:
{context}

Question:
{question}

Helpful Answer:
```

---

## 📌 Notes

* You can modify the embedding model by changing the model name in `HuggingFaceEmbeddings`.
* Replace static PDF path with `st.file_uploader()` to make it fully dynamic.
* Ensure your Groq API access includes the `gemma2-9b-it` model.

---

## 📜 License

MIT License

---

## 🤝 Acknowledgements

* [LangChain](https://github.com/langchain-ai/langchain)
* [HuggingFace](https://huggingface.co/)
* [Groq](https://groq.com/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Streamlit](https://streamlit.io/)

---




