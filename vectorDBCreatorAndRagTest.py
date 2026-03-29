import chromadb
from pymongo import MongoClient
import uuid
import time
from chromadb.utils import embedding_functions

print("🔄 Loading embedding model ONCE...")
embedding_function_global = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-base-en-v1.5",
    normalize_embeddings=True
)
print("✅ Model loaded once!")

class LocalLawVectorDB:
    def __init__(self, mongodb_uri="mongodb://localhost:27017/", db_name="pakistan_law_db"):
        print("🚀 Initializing Local Vector Database (No Downloads)...")
        start_time = time.time()
        
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        self.sections_collection = self.db["penal_code_sections"]
        
        # Initialize ChromaDB with Embedding Function
        self.chroma_client = chromadb.PersistentClient(path="./law_vector_db")
        
        embedding_function = embedding_function_global  # Use the global embedding function loaded once
        
        # Clear existing collection
        try:
            self.chroma_client.delete_collection("pakistan_penal_code")
            print("✅ Cleared existing collection")
        except:
            pass
            
        # Create collection with DEFAULT embedding function (no download)
        self.collection = self.chroma_client.get_or_create_collection(
            name="pakistan_penal_code",
            embedding_function=embedding_function
            ## Before Case of ONNIX ## No embedding function specified = uses default
        )
        
        print(f"✅ Initialization completed in {time.time() - start_time:.2f} seconds")
    
    # OLD method without chunking
    # def prepare_documents_simple(self):
    #     """Simple document preparation"""
    #     print("📚 Preparing documents from MongoDB...")
        
    #     documents = []
    #     metadatas = []
    #     ids = []
        
    #     sections = self.sections_collection.find({})
        
    #     for section in sections:
    #         section_id = str(section.get("section_number", ""))
    #         content = section.get("content", "").strip()
    #         title = section.get("section_title", "").strip()
    #         chapter = section.get("chapter", "")
            
    #         if not content:
    #             continue
            
    #         # Simple content - no chunking
    #         # if len(content) > 1500:
    #         #     content = content[:1500] + "..."
            
    #         full_content = f"Section {section_id}: {title}. {content}"
            
    #         documents.append(full_content)
    #         metadatas.append({
    #             "section_number": section_id,
    #             "section_title": title,
    #             "chapter": chapter,
    #             "source": "Pakistan Penal Code"
    #         })
    #         ids.append(str(uuid.uuid4())[:16])
        
    #     print(f"✅ Prepared {len(documents)} documents")
    #     return documents, metadatas, ids


    ### New method with chunking

    def prepare_documents_simple(self):
        print("📚 Preparing documents from MongoDB...")
    
        documents = []
        metadatas = []
        ids = []
        
        sections = self.sections_collection.find({})
        
        for section in sections:
            section_id = str(section.get("section_number", ""))
            content = section.get("content", "").strip()
            title = section.get("section_title", "").strip()
            chapter = section.get("chapter", "")
            
            if not content:
                continue
            
            # 🔥 ADD THIS: chunking
            chunks = self.chunk_text(content)

            for idx, chunk in enumerate(chunks):
                documents.append(f"Section {section_id}: {title}. {chunk}")
                
                metadatas.append({
                    "section_number": section_id,
                    "chunk_id": idx,
                    "section_title": title,
                    "chapter": chapter,
                    "source": "Pakistan Penal Code"
                })
                
                # ids.append(f"{section_id}_{idx}")
                ids.append(f"{section_id}_{idx}_{uuid.uuid4().hex[:6]}")
        
        print(f"✅ Prepared {len(documents)} chunks")
        return documents, metadatas, ids

    def chunk_text(self, text, chunk_size=300, overlap=50):
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def create_vector_database_fast(self):
        """Create vector database quickly"""
        print("🔄 Creating vector database...")
        total_start = time.time()
        
        documents, metadatas, ids = self.prepare_documents_simple()
        
        if not documents:
            print("❌ No documents to process!")
            return
        
        # Add in small batches
        batch_size = 100  # Adjust based on performance

        for i in range(0, len(documents), batch_size):
            batch_start = time.time()
            
            end_idx = min(i + batch_size, len(documents))
            batch_docs = documents[i:end_idx]
            batch_metas = metadatas[i:end_idx]
            batch_ids = ids[i:end_idx]
            
            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )
            
            batch_time = time.time() - batch_start
            print(f"✅ Batch {i//batch_size + 1} added ({len(batch_docs)} docs, {batch_time:.2f}s)")
        
        total_time = time.time() - total_start
        print(f"\n🎉 Database created in {total_time:.2f} seconds!")
        print(f"📊 Total documents: {self.collection.count()}")

def test_local_rag():
    """Test the local RAG system"""
    print("🧪 Testing Local RAG System...")
    
    ################################
    # Initialize and create database Everytime (for testing) 
    # - comment out if you want to skip creation after first run
    #################################

    vector_db = LocalLawVectorDB()
    vector_db.create_vector_database_fast()
    

    
    ################################
    # For testing, we can skip the creation step if the DB already exists to save time
    # - This allows us to test the search functionality without rebuilding the DB every time
    #################################
    # vector_db = LocalLawVectorDB()
    # if vector_db.collection.count() == 0:
    #     print("📦 Creating DB (first time only)...")
    #     vector_db.create_vector_database_fast()
    # else:
    #     print("⚡ Using existing vector DB (no rebuild)")

    # Test searches
    test_queries = [
        "murder punishment",
        "theft property", 
        "assault hurt",
        "fraud cheating"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Searching: '{query}'")
        try:
            results = vector_db.collection.query(
                query_texts=[query],
                n_results=10  # retrieve more
                )
            
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                print(f"   {i+1}. Section {metadata['section_number']}: {metadata['section_title']}")
                print(f"      Preview: {doc[:100]}...")
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}")

if __name__ == "__main__":
    test_local_rag()