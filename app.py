import streamlit as st
import os
from datetime import datetime
from data_ingestion import DataIngestion
from summarizer import Summarizer
from config import Config

# Page configuration
st.set_page_config(
    page_title="NJ Admin Code Summarizer",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F1F5F9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .summary-box {
        background-color: #F8FAFC;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #E2E8F0;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #F59E0B;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #FEE2E2;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #EF4444;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = None
if 'ingestion' not in st.session_state:
    st.session_state.ingestion = DataIngestion()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


def initialize_vectorstore():
    """Initialize or load vector store"""
    try:
        # Try to load existing vector store
        st.session_state.vectorstore = st.session_state.ingestion.load_vector_store()
        st.session_state.summarizer = Summarizer(st.session_state.vectorstore)
        return True, "Vector store loaded successfully!"
    except Exception as e:
        return False, f"No existing vector store found. Please upload a document first."


def process_uploaded_file(uploaded_file):
    """Process uploaded file and create vector store"""
    try:
        # Save uploaded file
        file_path = os.path.join(Config.DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process file
        with st.spinner("🔄 Processing document... This may take a minute..."):
            st.session_state.vectorstore = st.session_state.ingestion.process_file(
                file_path,
                save_vectorstore=True
            )
            st.session_state.summarizer = Summarizer(st.session_state.vectorstore)

        # Get stats
        stats = st.session_state.ingestion.get_stats(st.session_state.vectorstore)

        return True, stats
    except Exception as e:
        return False, str(e)


def display_summary_result(result):
    """Display summary result with formatting"""
    if result['success']:
        # Success header
        st.markdown('<div class="success-box">✅ <b>Summary Generated Successfully!</b></div>',
                    unsafe_allow_html=True)

        # Summary
        st.markdown("### 📝 Summary")
        st.markdown(f'<div class="summary-box">{result["summary"]}</div>',
                    unsafe_allow_html=True)

        # Download button
        st.download_button(
            label="📥 Download Summary",
            data=result['summary'],
            file_name=f"nj_admin_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

        # Statistics
        stats = result['metadata']['stats']
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Query Tokens", stats['query_tokens'])
        with col2:
            st.metric("Context Tokens", stats['context_tokens'])
        with col3:
            st.metric("Summary Tokens", stats['summary_tokens'])
        with col4:
            st.metric("Docs Retrieved", result['metadata']['retrieved_docs'])

        # Source citations
        with st.expander("📚 View Source Citations"):
            for i, source in enumerate(result['metadata']['sources'], 1):
                st.markdown(f"""
                <div class="source-box">
                    <b>Source {i}:</b><br>
                    <b>Title:</b> {source.get('title_name', 'N/A')}<br>
                    <b>Chapter:</b> {source.get('chapter_name', 'N/A')}<br>
                    <b>Section:</b> {source.get('section_name', 'N/A')}
                </div>
                """, unsafe_allow_html=True)

    else:
        # Error message
        st.markdown(f'<div class="error-box">{result["summary"]}</div>',
                    unsafe_allow_html=True)


# Main App
def main():
    # Header
    st.markdown('<div class="main-header">📋 NJ Admin Code Summarizer</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Document Summarization for New Jersey Administrative Code</div>',
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        # File upload section
        st.subheader("📂 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload your document",
            type=['csv', 'pdf', 'txt', 'docx'],
            help="Supported formats: CSV, PDF, TXT, DOCX"
        )

        if uploaded_file:
            if st.button("🚀 Process Document"):
                success, result = process_uploaded_file(uploaded_file)

                if success:
                    st.success("✅ Document processed successfully!")
                    st.json(result)
                else:
                    st.error(f"❌ Error: {result}")

        st.divider()

        # Load existing vector store
        st.subheader("💾 Load Existing Data")
        if st.button("📥 Load Vector Store"):
            success, message = initialize_vectorstore()
            if success:
                st.success(message)
            else:
                st.info(message)

        st.divider()

        # Settings
        st.subheader("🎛️ Settings")
        num_docs = st.slider(
            "Number of documents to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="More documents = more context but slower"
        )

        # Clear history
        if st.button("🗑️ Clear Chat History"):
            if st.session_state.summarizer:
                st.session_state.summarizer.clear_chat_history()
                st.session_state.chat_history = []
                st.success("Chat history cleared!")

        st.divider()

        # Info
        st.subheader("ℹ️ About")
        st.info("""
        This application uses RAG (Retrieval Augmented Generation) to summarize 
        New Jersey Administrative Code documents.

        **Features:**
        - Multi-format support
        - Domain relevance checking
        - Token management
        - Source citations
        - Chat history
        """)

    # Main content area
    if st.session_state.vectorstore is None:
        # Welcome screen
        st.info("👋 Welcome! Please upload a document or load an existing vector store from the sidebar to get started.")

        # Example queries
        st.markdown("### 📖 Example Queries")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Valid Queries:**
            - Summarize regulations about inmate correspondence
            - What are the rules for inmate visits?
            - Explain disciplinary procedures for inmates
            - Tell me about religious services for inmates
            """)

        with col2:
            st.markdown("""
            **Invalid Queries:**
            - Tell me about Python programming
            - What is machine learning?
            - How to cook pasta?

            *(Will be rejected as not relevant to NJ Admin Code)*
            """)

    else:
        # Query interface
        st.markdown("### 💬 Ask a Question")

        # Query input
        query = st.text_area(
            "Enter your question about NJ Admin Code:",
            height=100,
            placeholder="e.g., Summarize the regulations about inmate correspondence and mail..."
        )

        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.button("🔍 Summarize", type="primary", use_container_width=True)
        with col2:
            if st.button("❌ Exit Session", use_container_width=True):
                st.session_state.vectorstore = None
                st.session_state.summarizer = None
                st.rerun()

        if submit_button and query:
            if st.session_state.summarizer:
                # Generate summary
                with st.spinner("🤖 Generating summary..."):
                    result = st.session_state.summarizer.summarize(query, k=num_docs)

                # Display result
                display_summary_result(result)

                # Add to chat history
                if result['success']:
                    st.session_state.chat_history.append({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'query': query,
                        'summary': result['summary'][:200] + "..."
                    })
            else:
                st.error("❌ Summarizer not initialized. Please upload a document first.")

        # Chat history
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("### 📜 Chat History")

            for i, entry in enumerate(reversed(st.session_state.chat_history), 1):
                with st.expander(f"🕐 {entry['timestamp']} - Query {len(st.session_state.chat_history) - i + 1}"):
                    st.markdown(f"**Query:** {entry['query']}")
                    st.markdown(f"**Summary Preview:** {entry['summary']}")


if __name__ == "__main__":
    main()