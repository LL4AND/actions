import chromadb
import os
import sys

# Add project root to path to import from lpm_kernel
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lpm_kernel.api.services.user_llm_config_service import UserLLMConfigService
from lpm_kernel.file_data.chroma_utils import detect_embedding_model_dimension, reinitialize_chroma_collections

def init_chroma_db():
    chroma_path = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    
    # ensure the directory is correct
    os.makedirs(chroma_path, exist_ok=True)

    # Get embedding model dimension from user config
    try:
        user_llm_config_service = UserLLMConfigService()
        user_llm_config = user_llm_config_service.get_available_llm()
        
        if user_llm_config and user_llm_config.embedding_model_name:
            # Detect dimension based on model name
            dimension = detect_embedding_model_dimension(user_llm_config.embedding_model_name)
            print(f"Detected embedding dimension: {dimension} for model: {user_llm_config.embedding_model_name}")
        else:
            # Default to OpenAI dimension if no config found
            dimension = 1536
            print(f"No embedding model configured, using default dimension: {dimension}")
    except Exception as e:
        # Default to OpenAI dimension if error occurs
        dimension = 1536
        print(f"Error detecting embedding dimension, using default: {dimension}. Error: {e}")

    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collections_to_init = ["documents", "document_chunks"]
        dimension_mismatch_detected = False
        
        # collection: init documents level
        try:
            documents_collection = client.get_collection(name="documents")
            print(f"Collection 'documents' already exists")
        except ValueError:
            documents_collection = client.create_collection(
                name="documents",
                metadata={
                    "hnsw:space": "cosine",
                    "dimension": 1536
                }
            )
            print(f"Successfully created collection 'documents'")
            
        # collection: init chunk level
        try:
            chunks_collection = client.get_collection(name="document_chunks")
            print(f"Collection 'document_chunks' already exists")
        except ValueError:
            chunks_collection = client.create_collection(
                name="document_chunks",
                metadata={
                    "hnsw:space": "cosine",
                    "dimension": 1536
                }
            )
            print(f"Successfully created collection 'document_chunks'")
        
        print(f"ChromaDB initialized at {chroma_path}")
    except Exception as e:
        print(f"An error occurred while initializing ChromaDB: {e}")
        # no exception for following process
        # ChromaRepository will create collection if needed

if __name__ == "__main__":
    init_chroma_db()
