import os
import logging
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
    UnstructuredFileLoader
)

logger = logging.getLogger(__name__)


def load_document(file_path: str):
    """
    Load document using appropriate LangChain loader based on file extension
    
    Args:
        file_path: Path to the document file
    
    Returns:
        List of Document objects
    
    Raises:
        Exception: If loading fails
    """
    try:
        file_ext = os.path.splitext(file_path.lower())[1]
        
        logger.info(f"📄 Loading document: {os.path.basename(file_path)} (type: {file_ext})")
        
        # Select appropriate loader
        if file_ext == '.pdf':
            loader = PyPDFLoader(file_path)
            
        elif file_ext == '.txt':
            loader = TextLoader(file_path, encoding='utf-8')
            
        elif file_ext == '.docx':
            loader = Docx2txtLoader(file_path)
            
        elif file_ext == '.pptx':
            loader = UnstructuredPowerPointLoader(file_path)
            
        elif file_ext == '.csv':
            loader = CSVLoader(file_path, encoding='utf-8')
            
        else:
            # Fallback to unstructured loader
            logger.warning(f"⚠️  Using fallback loader for {file_ext}")
            loader = UnstructuredFileLoader(file_path)
        
        # Load documents
        documents = loader.load()
        
        if not documents:
            raise Exception("No content extracted from file")
        
        logger.info(f"✅ Loaded {len(documents)} document(s) from {os.path.basename(file_path)}")
        
        return documents
        
    except Exception as e:
        logger.error(f"❌ Failed to load {file_path}: {e}")
        raise Exception(f"Failed to load document: {str(e)}")