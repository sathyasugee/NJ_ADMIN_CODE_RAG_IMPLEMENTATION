import os

from dotenv import load_dotenv

load_dotenv('.env')

class Config:
    """ configuration for NJ_admin_Code """

    #config API KEY
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    #llm Model Configuration
    LLM_MODEL = "llama-3.3-70b-versatile"
    TEMPERATURE = 0

    #config Embedding Model
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    #chuncking configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    #config the vector store
    VECTOR_STORE_PATH ="./vector_store"

    #token limits
    MAX_INPUT_TOKENS = 6000
    MAX_OUTPUT_TOKENS = 1024

    #Relavant Key word for NJ_document
    DOMAIN_KEYWORDS = [
        "new jersey", "nj", "corrections", "inmate", "correctional facility",
        "department of corrections", "administrator", "commissioner", "njac",
        "regulation", "statute", "policy", "procedure", "chapter", "subchapter"
    ]

    #Config the summary style
    SUMMARY_STYLE = "concise"

    # Supported File Formats
    SUPPORTED_FORMATS = [".pdf", ".txt", ".csv", ".docx"]

    # Data Directory
    DATA_DIR = "./data_dir"

    #chat history
    MAX_CHAT_HISTORY = 5

    @staticmethod
    def validate():
        """Validate configuration"""
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not found in .env file!")

        if not os.path.exists(Config.DATA_DIR):
            os.makedirs(Config.DATA_DIR)
            print(f"Created data directory: {Config.DATA_DIR}")

        if not os.path.exists(Config.VECTOR_STORE_PATH):
            os.makedirs(Config.VECTOR_STORE_PATH)
            print(f"Created vector store directory: {Config.VECTOR_STORE_PATH}")

        print("✅ Configuration validated successfully!")
        return True


