# 📋 NJ Admin Code Summarizer (RAG-based Streamlit App)

An AI-powered Streamlit application that summarizes New Jersey Administrative Code documents using **Retrieval-Augmented Generation (RAG)**.  
This app allows you to upload documents, generate concise summaries, and explore source citations — all through an intuitive web interface.

---

## 🚀 Features

- 📄 Upload and process documents (`PDF`, `CSV`, `TXT`, `DOCX`)
- 🔍 Retrieve and summarize sections of NJ Administrative Code using RAG
- 💾 Persistent vector store for faster future queries
- 📊 Interactive Streamlit interface with chat history
- ⚙️ Environment-based configuration for easy setup and deployment

---

## 🗂️ Project Structure

| File / Folder | Description |
|----------------|--------------|
| **`app.py`** | The main Streamlit app that runs the user interface. Handles document upload, summarization queries, and displaying summary results interactively. |
| **`config.py`** | Contains configuration settings such as API keys, file paths, and environment variables. Ensures consistent access to configurations throughout the app. |
| **`data_ingestion.py`** | Handles document ingestion, text splitting, embedding creation, and vector storage. Builds the knowledge base used by the summarizer. |
| **`summarizer.py`** | Implements the summarization logic using the vector store and LLM. Retrieves relevant context and generates concise summaries for user queries. |
| **`requirements.txt`** | Lists all Python dependencies required to run the project. Use it to install all necessary packages with one command. |
| **`.env_template.txt`** | Template for environment variables like API keys and data directory paths. Copy it to `.env` and fill in your own configuration values. |
| **`README.md`** | Provides a complete overview of the project, setup instructions, and explanations for each file. Helps others understand and run your app easily. |

---

## ⚙️ Setup Instructions

1. **Clone this repository**
   ```bash
   git clone https://github.com/<your-username>/nj_rag_streamlit_implementation.git
   cd nj_rag_streamlit_implementation
