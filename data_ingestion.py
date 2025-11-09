import os
import pandas as pd
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import Config


class DataIngestion:
    """
    Class to handle document loading, chunking, and vector store creation
    for NJ Admin Code Summarization
    """

    def __init__(self):
        """Initialize DataIngestion with configuration"""
        self.config = Config()
        self.config.validate()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.EMBEDDING_MODEL
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len
        )

    def load_csv(self, file_path: str) -> List[Document]:
        """
        Load CSV file and convert to LangChain Documents with metadata

        Args:
            file_path: Path to CSV file

        Returns:
            List of Document objects with content and metadata
        """
        try:
            df = pd.read_csv(file_path)
            documents = []

            for idx, row in df.iterrows():
                # Create content with structure (title + chapter + content)
                content = f"""
Title: {row.get('title_name', 'N/A')}
Chapter: {row.get('chapter_name', 'N/A')}
Section: {row.get('section_name', 'N/A')}

Content:
{row.get('section_content_text', '')}
                """.strip()

                # Create metadata
                metadata = {
                    'source': file_path,
                    'row_index': idx,
                    'primary_key': row.get('primary_key', 'N/A'),
                    'title_num': row.get('title_num', 'N/A'),
                    'chapter_num': row.get('chapter_num', 'N/A'),
                    'section_num': row.get('section_num', 'N/A'),
                    'title_name': row.get('title_name', 'N/A'),
                    'chapter_name': row.get('chapter_name', 'N/A'),
                    'section_name': row.get('section_name', 'N/A')
                }

                documents.append(Document(page_content=content, metadata=metadata))

            print(f"✅ Loaded {len(documents)} documents from CSV")
            return documents

        except Exception as e:
            print(f"❌ Error loading CSV: {str(e)}")
            raise

    def load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF file"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            print(f"✅ Loaded {len(documents)} pages from PDF")
            return documents
        except Exception as e:
            print(f"❌ Error loading PDF: {str(e)}")
            raise

    def load_text(self, file_path: str) -> List[Document]:
        """Load TXT file"""
        try:
            loader = TextLoader(file_path)
            documents = loader.load()
            print(f"✅ Loaded text file")
            return documents
        except Exception as e:
            print(f"❌ Error loading TXT: {str(e)}")
            raise

    def load_docx(self, file_path: str) -> List[Document]:
        """Load DOCX file"""
        try:
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
            print(f"✅ Loaded DOCX file")
            return documents
        except Exception as e:
            print(f"❌ Error loading DOCX: {str(e)}")
            raise

    def load_documents(self, file_path: str) -> List[Document]:
        """
        Load documents based on file extension

        Args:
            file_path: Path to the file

        Returns:
            List of Document objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()

        if file_ext not in self.config.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported formats: {self.config.SUPPORTED_FORMATS}"
            )

        print(f"📂 Loading file: {file_path}")

        if file_ext == '.csv':
            return self.load_csv(file_path)
        elif file_ext == '.pdf':
            return self.load_pdf(file_path)
        elif file_ext == '.txt':
            return self.load_text(file_path)
        elif file_ext == '.docx':
            return self.load_docx(file_path)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks

        Args:
            documents: List of Document objects

        Returns:
            List of chunked Document objects
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            print(f"✅ Created {len(chunks)} chunks from {len(documents)} documents")
            return chunks
        except Exception as e:
            print(f"❌ Error chunking documents: {str(e)}")
            raise

    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """
        Create FAISS vector store from document chunks

        Args:
            chunks: List of chunked Document objects

        Returns:
            FAISS vector store
        """
        try:
            print("🔄 Creating vector embeddings... (this may take a moment)")
            vectorstore = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            print("✅ Vector store created successfully!")
            return vectorstore
        except Exception as e:
            print(f"❌ Error creating vector store: {str(e)}")
            raise

    def save_vector_store(self, vectorstore: FAISS, path: str = None):
        """
        Save vector store to disk

        Args:
            vectorstore: FAISS vector store
            path: Path to save (default: config.VECTOR_STORE_PATH)
        """
        try:
            save_path = path or self.config.VECTOR_STORE_PATH
            vectorstore.save_local(save_path)
            print(f"✅ Vector store saved to: {save_path}")
        except Exception as e:
            print(f"❌ Error saving vector store: {str(e)}")
            raise

    def load_vector_store(self, path: str = None) -> FAISS:
        """
        Load existing vector store from disk

        Args:
            path: Path to load from (default: config.VECTOR_STORE_PATH)

        Returns:
            FAISS vector store
        """
        try:
            load_path = path or self.config.VECTOR_STORE_PATH

            if not os.path.exists(load_path):
                raise FileNotFoundError(f"Vector store not found at: {load_path}")

            vectorstore = FAISS.load_local(
                load_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✅ Vector store loaded from: {load_path}")
            return vectorstore
        except Exception as e:
            print(f"❌ Error loading vector store: {str(e)}")
            raise

    def process_file(self, file_path: str, save_vectorstore: bool = True) -> FAISS:
        """
        Complete pipeline: Load → Chunk → Create Vector Store

        Args:
            file_path: Path to the file to process
            save_vectorstore: Whether to save the vector store to disk

        Returns:
            FAISS vector store
        """
        print("\n" + "=" * 60)
        print("🚀 Starting Data Ingestion Pipeline")
        print("=" * 60)

        # Step 1: Load documents
        documents = self.load_documents(file_path)

        # Step 2: Chunk documents
        chunks = self.chunk_documents(documents)

        # Step 3: Create vector store
        vectorstore = self.create_vector_store(chunks)

        # Step 4: Save vector store (optional)
        if save_vectorstore:
            self.save_vector_store(vectorstore)

        print("=" * 60)
        print("✅ Data Ingestion Pipeline Completed Successfully!")
        print("=" * 60 + "\n")

        return vectorstore

    def get_stats(self, vectorstore: FAISS) -> Dict[str, Any]:
        """
        Get statistics about the vector store

        Args:
            vectorstore: FAISS vector store

        Returns:
            Dictionary with statistics
        """
        try:
            index = vectorstore.index
            stats = {
                'total_vectors': index.ntotal,
                'vector_dimension': index.d,
                'is_trained': index.is_trained
            }
            return stats
        except Exception as e:
            print(f"❌ Error getting stats: {str(e)}")
            return {}


# Test the DataIngestion class
if __name__ == "__main__":
    # Initialize
    ingestion = DataIngestion()

    # Process your CSV file
    csv_path = "./data_dir/nj_rag_implementation.csv"

    if os.path.exists(csv_path):
        # Process and create vector store
        vectorstore = ingestion.process_file(csv_path, save_vectorstore=True)

        # Print statistics
        stats = ingestion.get_stats(vectorstore)
        print("\n📊 Vector Store Statistics:")
        print(f"   Total Vectors: {stats.get('total_vectors', 'N/A')}")
        print(f"   Vector Dimension: {stats.get('vector_dimension', 'N/A')}")
        print(f"   Is Trained: {stats.get('is_trained', 'N/A')}")

        # Test retrieval
        print("\n🔍 Testing Retrieval:")
        test_query = "What are the regulations about inmate correspondence?"
        docs = vectorstore.similarity_search(test_query, k=3)
        print(f"   Query: {test_query}")
        print(f"   Retrieved {len(docs)} relevant documents")
        print(f"   First result: {docs[0].page_content[:200]}...")
    else:
        print(f"❌ File not found: {csv_path}")
        print("Please place your CSV file in the data_dir folder")