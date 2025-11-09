import tiktoken
from typing import List, Dict, Tuple
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from config import Config


class Summarizer:
    """
    Class to handle document summarization with relevance checking,
    token management, and chat history
    """

    def __init__(self, vectorstore: FAISS):
        """
        Initialize Summarizer

        Args:
            vectorstore: FAISS vector store with embedded documents
        """
        self.config = Config()
        self.vectorstore = vectorstore
        self.llm = ChatGroq(
            model=self.config.LLM_MODEL,
            temperature=self.config.TEMPERATURE,
            groq_api_key=self.config.GROQ_API_KEY
        )
        self.chat_history = []
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            print(f"⚠️ Token counting error: {str(e)}")
            # Fallback: rough estimate (1 token ≈ 4 characters)
            return len(text) // 4

    def check_relevance(self, query: str) -> Tuple[bool, str]:
        """
        Check if query is relevant to NJ Admin Code domain

        Args:
            query: User query

        Returns:
            Tuple of (is_relevant: bool, message: str)
        """
        query_lower = query.lower()

        # Check for domain keywords
        has_domain_keyword = any(
            keyword in query_lower
            for keyword in self.config.DOMAIN_KEYWORDS
        )

        # If no domain keywords, do a quick retrieval test
        if not has_domain_keyword:
            try:
                # Retrieve top 1 document
                docs = self.vectorstore.similarity_search(query, k=1)
                if docs:
                    # Check if retrieved doc has domain indicators
                    doc_text = docs[0].page_content.lower()
                    has_domain_in_doc = any(
                        keyword in doc_text
                        for keyword in self.config.DOMAIN_KEYWORDS
                    )

                    if not has_domain_in_doc:
                        return False, (
                            "❌ **Query Not Relevant**\n\n"
                            "I can only summarize content related to **New Jersey Administrative Code** "
                            "(NJ Admin Code for Corrections).\n\n"
                            "Your query doesn't seem to be related to this domain.\n\n"
                            "**Example relevant queries:**\n"
                            "- 'Summarize regulations about inmate correspondence'\n"
                            "- 'What are the rules for inmate visits?'\n"
                            "- 'Explain the disciplinary procedures for inmates'\n"
                        )
            except Exception as e:
                print(f"⚠️ Relevance check error: {str(e)}")

        return True, "Query is relevant"

    def retrieve_documents(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant documents from vector store

        Args:
            query: User query
            k: Number of documents to retrieve

        Returns:
            List of documents with content and metadata
        """
        try:
            docs = self.vectorstore.similarity_search(query, k=k)

            retrieved_docs = []
            for doc in docs:
                retrieved_docs.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })

            print(f"📥 Retrieved {len(retrieved_docs)} relevant documents")
            return retrieved_docs

        except Exception as e:
            print(f"❌ Error retrieving documents: {str(e)}")
            return []

    def prepare_context(self, documents: List[Dict], max_tokens: int = None) -> str:
        """
        Prepare context from retrieved documents within token limit

        Args:
            documents: List of retrieved documents
            max_tokens: Maximum tokens for context (default: config limit)

        Returns:
            Combined context string
        """
        max_tokens = max_tokens or self.config.MAX_INPUT_TOKENS

        context_parts = []
        current_tokens = 0

        for i, doc in enumerate(documents):
            doc_content = doc['content']
            doc_tokens = self.count_tokens(doc_content)

            # Check if adding this document exceeds limit
            if current_tokens + doc_tokens > max_tokens:
                print(f"⚠️ Token limit reached. Using {i} out of {len(documents)} documents")
                break

            # Add section metadata for citation
            metadata = doc['metadata']
            section_info = (
                f"\n--- Document {i + 1} ---\n"
                f"Title: {metadata.get('title_name', 'N/A')}\n"
                f"Chapter: {metadata.get('chapter_name', 'N/A')}\n"
                f"Section: {metadata.get('section_name', 'N/A')}\n\n"
                f"{doc_content}\n"
            )

            context_parts.append(section_info)
            current_tokens += doc_tokens

        context = "\n".join(context_parts)
        print(f"📊 Context prepared: {current_tokens} tokens from {len(context_parts)} documents")

        return context

    def create_summary_prompt(self, query: str, context: str) -> ChatPromptTemplate:
        """
        Create prompt template for summarization

        Args:
            query: User query
            context: Retrieved context

        Returns:
            ChatPromptTemplate
        """
        template = """You are an expert assistant specializing in New Jersey Administrative Code (NJ Admin Code) for Corrections.

Your task is to provide a clear, accurate, and concise summary based on the retrieved documents.

**Context from NJ Admin Code:**
{context}

**User Query:**
{query}

**Instructions:**
1. Summarize the relevant information from the context that answers the user's query
2. Be accurate - only include information present in the context
3. Organize the summary in a clear, structured format
4. If specific regulations or procedures are mentioned, include section references
5. Keep the summary concise but comprehensive (aim for 150-300 words)
6. If the context doesn't contain relevant information, state that clearly

**Summary:**"""

        prompt = ChatPromptTemplate.from_template(template)
        return prompt

    def generate_summary(self, query: str, context: str) -> str:
        """
        Generate summary using LLM

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Generated summary
        """
        try:
            # Create prompt
            prompt = self.create_summary_prompt(query, context)

            # Create chain
            chain = prompt | self.llm

            # Generate summary
            print("🤖 Generating summary...")
            response = chain.invoke({
                "query": query,
                "context": context
            })

            summary = response.content
            print("✅ Summary generated successfully!")

            return summary

        except Exception as e:
            print(f"❌ Error generating summary: {str(e)}")
            return f"Error generating summary: {str(e)}"

    def add_to_chat_history(self, query: str, summary: str):
        """
        Add query and summary to chat history (memory efficient)

        Args:
            query: User query
            summary: Generated summary
        """
        # Store only essential info
        self.chat_history.append({
            'query': query[:200],  # Truncate long queries
            'summary': summary[:500],  # Truncate long summaries
            'timestamp': self._get_timestamp()
        })

        # Keep only last N conversations
        if len(self.chat_history) > self.config.MAX_CHAT_HISTORY:
            self.chat_history = self.chat_history[-self.config.MAX_CHAT_HISTORY:]

        print(f"💾 Chat history updated (size: {len(self.chat_history)})")

    def get_chat_history(self) -> List[Dict]:
        """Get current chat history"""
        return self.chat_history

    def clear_chat_history(self):
        """Clear chat history"""
        self.chat_history = []
        print("🗑️ Chat history cleared")

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_summary_stats(self, query: str, summary: str, context: str) -> Dict:
        """
        Get statistics about the summarization

        Args:
            query: User query
            summary: Generated summary
            context: Retrieved context

        Returns:
            Dictionary with statistics
        """
        return {
            'query_tokens': self.count_tokens(query),
            'context_tokens': self.count_tokens(context),
            'summary_tokens': self.count_tokens(summary),
            'summary_length': len(summary),
            'chat_history_size': len(self.chat_history)
        }

    def summarize(self, query: str, k: int = 5) -> Dict:
        """
        Complete summarization pipeline

        Args:
            query: User query
            k: Number of documents to retrieve

        Returns:
            Dictionary with summary and metadata
        """
        print("\n" + "=" * 60)
        print("🚀 Starting Summarization Pipeline")
        print("=" * 60)

        # Step 1: Check relevance
        print("\n1️⃣ Checking query relevance...")
        is_relevant, message = self.check_relevance(query)

        if not is_relevant:
            print("❌ Query not relevant to NJ Admin Code domain")
            return {
                'success': False,
                'summary': message,
                'metadata': {
                    'error': 'Query not relevant'
                }
            }

        print("✅ Query is relevant")

        # Step 2: Retrieve documents
        print("\n2️⃣ Retrieving relevant documents...")
        documents = self.retrieve_documents(query, k=k)

        if not documents:
            return {
                'success': False,
                'summary': "❌ No relevant documents found for your query.",
                'metadata': {
                    'error': 'No documents retrieved'
                }
            }

        # Step 3: Prepare context
        print("\n3️⃣ Preparing context...")
        context = self.prepare_context(documents)

        # Step 4: Generate summary
        print("\n4️⃣ Generating summary...")
        summary = self.generate_summary(query, context)

        # Step 5: Get statistics
        stats = self.get_summary_stats(query, summary, context)

        # Step 6: Add to chat history
        self.add_to_chat_history(query, summary)

        print("\n" + "=" * 60)
        print("✅ Summarization Pipeline Completed!")
        print("=" * 60 + "\n")

        return {
            'success': True,
            'summary': summary,
            'metadata': {
                'retrieved_docs': len(documents),
                'stats': stats,
                'sources': [doc['metadata'] for doc in documents]
            }
        }


# Test the Summarizer class
if __name__ == "__main__":
    from data_ingestion import DataIngestion

    print("🧪 Testing Summarizer...")

    # Load vector store
    ingestion = DataIngestion()

    try:
        vectorstore = ingestion.load_vector_store()
        print("✅ Vector store loaded\n")
    except Exception as e:
        print(f"❌ Vector store not found. Running data ingestion first...\n")
        csv_path = "./data_dir/nj_rag_implementation.csv"
        vectorstore = ingestion.process_file(csv_path)

    # Initialize summarizer
    summarizer = Summarizer(vectorstore)

    # Test queries
    test_queries = [
        "Summarize the regulations about inmate correspondence and mail",
        "What are the rules for inmate visits?",
        "Tell me about Python programming"  # Should be rejected
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'=' * 60}")
        print(f"TEST {i}: {query}")
        print('=' * 60)

        result = summarizer.summarize(query, k=3)

        if result['success']:
            print(f"\n📝 **SUMMARY:**\n{result['summary']}")
            print(f"\n📊 **STATS:**")
            stats = result['metadata']['stats']
            print(f"   Query Tokens: {stats['query_tokens']}")
            print(f"   Context Tokens: {stats['context_tokens']}")
            print(f"   Summary Tokens: {stats['summary_tokens']}")
            print(f"   Retrieved Docs: {result['metadata']['retrieved_docs']}")
        else:
            print(f"\n{result['summary']}")

        print("\n" + "=" * 60)

    # Test chat history
    print("\n📜 Chat History:")
    for i, entry in enumerate(summarizer.get_chat_history(), 1):
        print(f"\n{i}. {entry['timestamp']}")
        print(f"   Query: {entry['query'][:100]}...")
        print(f"   Summary: {entry['summary'][:100]}...")