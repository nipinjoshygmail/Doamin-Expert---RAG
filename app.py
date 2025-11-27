import os
import logging
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import chromadb
from werkzeug.utils import secure_filename
from utils.transcription import transcribe_audio
from utils.loaders import load_document

# Configure logging
os.makedirs('./logs', exist_ok=True)
log_file = './logs/rag_app.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"📝 Logging to file: {os.path.abspath(log_file)}")

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = './uploads'

# Ensure directories exist
os.makedirs('./uploads', exist_ok=True)
os.makedirs('./chroma_db', exist_ok=True)

# Supported file extensions
ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.pptx', '.csv', '.mp3', '.wav'}

# ==================== GLOBAL INITIALIZATION ====================
logger.info("🚀 Initializing RAG system...")

# Initialize embeddings model with better model
logger.info("📦 Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # Better model
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Initialize ChromaDB
logger.info("🗄️ Initializing ChromaDB...")
chroma_client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=chromadb.Settings(anonymized_telemetry=False)
)

# Get or create collection
COLLECTION_NAME = "rag_collection"
try:
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    logger.info(f"✅ Loaded existing collection: {COLLECTION_NAME}")
except Exception:
    collection = chroma_client.create_collection(name=COLLECTION_NAME)
    logger.info(f"✨ Created new collection: {COLLECTION_NAME}")

# Initialize vector store
vectorstore = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings
)

# Initialize retriever with improved settings
retriever = vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance for diversity
    search_kwargs={
        "k": 10,  # Retrieve more chunks
        "fetch_k": 20,  # Fetch more candidates for MMR
        "lambda_mult": 0.5  # Balance between relevance and diversity
    }
)

# Initialize text splitter with better settings
logger.info("✂️ Initializing text splitter...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,  # Smaller chunks for better granularity
    chunk_overlap=150,  # Good overlap to preserve context
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]  # Better semantic splitting
)

# Initialize LLM with Ollama
logger.info("🤖 Initializing LLM with Ollama...")
ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

try:
    llm = ChatOllama(
        base_url=ollama_base_url,
        model=ollama_model,
        temperature=0.2,  # Lower temperature for more focused answers
        num_ctx=4096  # Increase context window
    )
    logger.info(f"✅ Using Ollama model: {ollama_model} at {ollama_base_url}")
except Exception as e:
    logger.warning(f"⚠️ Failed to connect to Ollama: {e}")
    llm = None

# Improved prompt template with better instructions
prompt_template = """You are a helpful technical documentation assistant. Your goal is to provide accurate, detailed answers based on the provided context.

Context Information:
{context}

User Question: {question}

Instructions:
1. Answer the question using ONLY the information from the context above
2. If the context contains partial information, provide what you can and indicate what's missing
3. If the context doesn't contain relevant information, say: "I don't have information about this in the current knowledge base."
4. Be specific and cite relevant details from the context
5. If there are step-by-step instructions in the context, preserve that structure in your answer

Answer:"""

prompt = ChatPromptTemplate.from_template(prompt_template)

# Helper function for query expansion
def expand_query(question: str) -> list:
    """Generate alternative phrasings of the query for better retrieval"""
    expansions = [question]
    
    # Add keyword extraction
    keywords = question.lower().split()
    important_words = [w for w in keywords if len(w) > 3 and w not in {
        'what', 'where', 'when', 'how', 'why', 'the', 'this', 'that', 'with'
    }]
    
    if important_words:
        expansions.append(" ".join(important_words))
    
    return expansions

# Custom retriever with hybrid search
def hybrid_retrieve(question: str, k: int = 10):
    """Perform hybrid retrieval with query expansion"""
    all_docs = []
    seen_content = set()
    
    # Get queries
    queries = expand_query(question)
    
    for query in queries:
        try:
            docs = retriever.get_relevant_documents(query)
            for doc in docs:
                content_hash = hash(doc.page_content)
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)
        except Exception as e:
            logger.warning(f"Error retrieving for query '{query}': {e}")
    
    # Return top k unique documents
    return all_docs[:k]

# Create RAG chain with custom retriever
if llm:
    def format_docs(docs):
        """Format documents with source attribution"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            formatted.append(f"[Source {i}: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(formatted)
    
    rag_chain = (
        {"context": lambda x: format_docs(hybrid_retrieve(x)), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    logger.info("✅ RAG chain initialized successfully")
else:
    rag_chain = None

logger.info("🎉 RAG system ready!")

# ==================== HELPER FUNCTIONS ====================

def allowed_file(filename):
    """Check if file extension is allowed"""
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS


def clean_text(text):
    """Clean and normalize text while preserving structure"""
    import re
    # Remove excessive whitespace but preserve paragraphs
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove common noise
    text = re.sub(r'Page \d+ of \d+', '', text)
    text = re.sub(r'\[?\d+\]?', '', text)  # Remove page numbers/footnotes
    return text.strip()


def add_metadata_to_chunks(chunks, filename):
    """Add rich metadata to chunks for better retrieval"""
    for i, chunk in enumerate(chunks):
        chunk.metadata['source'] = filename
        chunk.metadata['chunk_id'] = i
        chunk.metadata['chunk_size'] = len(chunk.page_content)
        # Add first few words as preview
        preview = ' '.join(chunk.page_content.split()[:10])
        chunk.metadata['preview'] = preview
    return chunks


# ==================== ROUTES ====================

@app.route('/')
def index():
    """Home page - query interface"""
    return render_template('index.html')

@app.route('/documents')
def documents():
    """Documents visualization page"""
    return render_template('documents.html')


@app.route('/upload')
def upload_page():
    """File upload page"""
    return render_template('upload.html')


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "collection": COLLECTION_NAME,
        "total_chunks": collection.count()
    })


@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file upload and indexing"""
    try:
        import time
        start_time = time.time()
        logger.info("📤 Processing new file upload request...")
        
        if 'files' not in request.files:
            logger.warning("❌ No files found in request")
            return jsonify({"status": "error", "message": "No files provided"}), 400
        
        files = request.files.getlist('files')
        logger.info(f"📁 Received {len(files)} files: {[f.filename for f in files]}")
        
        if not files or all(f.filename == '' for f in files):
            logger.warning("❌ No valid files selected")
            return jsonify({"status": "error", "message": "No files selected"}), 400
        
        total_indexed = 0
        processed_files = []
        errors = []
        
        for file in files:
            if file.filename == '':
                continue
            
            filename = secure_filename(file.filename)
            file_ext = os.path.splitext(filename.lower())[1]
            
            if file_ext not in ALLOWED_EXTENSIONS:
                errors.append(f"{filename}: Unsupported file type")
                continue
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"📄 Processing: {filename}")
            
            try:
                # Handle audio files
                if file_ext in {'.mp3', '.wav'}:
                    logger.info(f"🎤 Transcribing audio: {filename}")
                    transcription = transcribe_audio(filepath)
                    txt_filename = f"{os.path.splitext(filename)[0]}_transcription.txt"
                    txt_filepath = os.path.join(app.config['UPLOAD_FOLDER'], txt_filename)
                    
                    with open(txt_filepath, 'w', encoding='utf-8') as f:
                        f.write(transcription)
                    
                    documents = load_document(txt_filepath)
                    os.remove(txt_filepath)
                else:
                    documents = load_document(filepath)
                
                if not documents:
                    errors.append(f"{filename}: No content extracted")
                    continue
                
                # Clean and prepare documents
                for doc in documents:
                    doc.page_content = clean_text(doc.page_content)
                    doc.metadata['source'] = filename
                
                # Split with better chunking
                logger.info(f"✂️ Splitting {filename} into chunks...")
                chunks = text_splitter.split_documents(documents)
                
                if not chunks:
                    errors.append(f"{filename}: No chunks created")
                    continue
                
                # Add rich metadata
                chunks = add_metadata_to_chunks(chunks, filename)
                
                logger.info(f"📊 Created {len(chunks)} chunks from {filename}")
                
                # Log sample chunks
                for i, chunk in enumerate(chunks[:2]):
                    logger.info(f"  Sample Chunk {i+1}:")
                    logger.info(f"    Preview: {chunk.metadata.get('preview', 'N/A')}")
                    logger.info(f"    Size: {len(chunk.page_content)} chars")
                
                # Generate unique IDs
                chunk_ids = [f"{filename}_{i}_{hash(chunk.page_content)}" for i in range(len(chunks))]
                
                # Add to vector store
                logger.info(f"💾 Adding chunks to vector store...")
                vectorstore.add_documents(documents=chunks, ids=chunk_ids)
                
                total_indexed += len(chunks)
                processed_files.append(filename)
                logger.info(f"✅ Indexed {len(chunks)} chunks from {filename}")
                
            except Exception as e:
                error_msg = f"{filename}: {str(e)}"
                errors.append(error_msg)
                logger.error(f"❌ Error processing {filename}: {e}")
            
            finally:
                if os.path.exists(filepath):
                    os.remove(filepath)
        
        if total_indexed == 0:
            return jsonify({
                "status": "error",
                "message": "No files were successfully processed",
                "errors": errors
            }), 400
        
        response = {
            "status": "success",
            "indexed": total_indexed,
            "files_processed": len(processed_files),
            "message": f"Successfully processed {len(processed_files)} file(s) and indexed {total_indexed} chunks"
        }
        
        if errors:
            response["warnings"] = errors
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"❌ Upload error: {e}")
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500


@app.route('/api/query', methods=['POST'])
def query():
    """Handle user queries with improved retrieval"""
    try:
        import time
        start_time = time.time()
        
        if not rag_chain:
            return jsonify({
                "status": "error",
                "message": "LLM not configured. Please ensure Ollama is running."
            }), 500
        
        data = request.get_json()
        logger.info(f"📥 Received request data: {data}")
        
        if not data or 'question' not in data:
            return jsonify({"status": "error", "message": "No question provided"}), 400
        
        question = data['question'].strip()
        
        if not question:
            return jsonify({"status": "error", "message": "Question cannot be empty"}), 400
        
        logger.info(f"❓ Processing query: {question}")
        
        # Check collection
        collection_count = collection.count()
        if collection_count == 0:
            return jsonify({
                "status": "error",
                "message": "No documents in knowledge base. Please upload files first."
            }), 400
        
        # Retrieve with hybrid approach
        retrieval_start = time.time()
        logger.info(f"🔍 Retrieving from {collection_count} total chunks using hybrid search...")
        
        relevant_docs = hybrid_retrieve(question, k=10)
        retrieval_time = time.time() - retrieval_start
        
        # Log retrieved chunks with relevance info
        logger.info(f"📚 Retrieved {len(relevant_docs)} relevant chunks in {retrieval_time:.2f}s:")
        for i, doc in enumerate(relevant_docs[:5], 1):  # Log top 5
            logger.info(f"  Chunk {i}:")
            logger.info(f"    Source: {doc.metadata.get('source', 'Unknown')}")
            logger.info(f"    Preview: {doc.metadata.get('preview', doc.page_content[:100])}")
        
        # Execute RAG chain
        llm_start = time.time()
        logger.info("🤖 Generating answer with LLM...")
        answer = rag_chain.invoke(question)
        llm_time = time.time() - llm_start
        
        logger.info(f"✅ Answer generated in {llm_time:.2f}s")
        logger.info(f"📤 Response: {answer[:200]}...")
        
        total_time = time.time() - start_time
        logger.info(f"⏱️ Total time: {total_time:.2f}s (Retrieval: {retrieval_time:.2f}s, LLM: {llm_time:.2f}s)")
        
        return jsonify({
            "status": "success",
            "answer": answer.strip(),
            "question": question,
            "metadata": {
                "chunks_retrieved": len(relevant_docs),
                "processing_time": total_time,
                "retrieval_time": retrieval_time,
                "llm_time": llm_time,
                "sources": list(set(doc.metadata.get('source', 'Unknown') for doc in relevant_docs))
            }
        }), 200
        
    except Exception as e:
        logger.error(f"❌ Query error: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": f"Error processing query: {str(e)}"
        }), 500


# ==================== DOCUMENT MANAGEMENT ====================

@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all documents and their chunks in the vector store"""
    try:
        results = collection.get(include=["metadatas"])
        
        documents = {}
        if results and results['metadatas']:
            for metadata in results['metadatas']:
                source = metadata.get('source')
                if not source:
                    continue
                
                if source not in documents:
                    documents[source] = {
                        'total_chunks': 0,
                        'total_size': 0,
                        'file_type': os.path.splitext(source)[1].lower()
                    }
                
                documents[source]['total_chunks'] += 1
                documents[source]['total_size'] += metadata.get('chunk_size', 0)

        document_list = [
            {
                'id': filename,
                'filename': filename,
                'type': info['file_type'],
                'size': info['total_size'],
                'chunks': info['total_chunks']
            }
            for filename, info in documents.items()
        ]
        
        return jsonify({
            'status': 'success',
            'documents': sorted(document_list, key=lambda x: x['filename'].lower())
        })
    
    except Exception as e:
        logger.error(f"❌ Error listing documents: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/documents/<path:doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    """Delete a document and all its chunks from the vector store"""
    try:
        if not doc_id:
            return jsonify({'status': 'error', 'message': 'Document ID is required'}), 400

        logger.info(f"🗑️ Request to delete document: {doc_id}")

        # Find all chunk IDs associated with this document
        results = collection.get(where={"source": doc_id}, include=["metadatas"])
        
        ids_to_delete = results.get('ids')

        if not ids_to_delete:
            logger.warning(f"⚠️ No chunks found for document: {doc_id}. It may have been already deleted.")
            return jsonify({'status': 'error', 'message': 'Document not found'}), 404

        # Delete the chunks from the collection
        collection.delete(ids=ids_to_delete)
        
        logger.info(f"✅ Successfully deleted {len(ids_to_delete)} chunks for document: {doc_id}")
        
        return jsonify({
            'status': 'success',
            'message': f'Successfully deleted document "{doc_id}"'
        })

    except Exception as e:
        logger.error(f"❌ Error deleting document {doc_id}: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ==================== RUN APP ====================

if __name__ == '__main__':
    logger.info("🌐 Starting Flask server...")
    app.run(debug=True, host='0.0.0.0', port=5000)
