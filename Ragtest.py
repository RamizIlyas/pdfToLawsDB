import chromadb
from pymongo import MongoClient
import uuid
import time

class LocalLawVectorDB:
    def __init__(self, mongodb_uri="mongodb://localhost:27017/", db_name="pakistan_law_db"):
        print("🚀 Initializing Local Vector Database (No Downloads)...")
        start_time = time.time()
        
        self.client = MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        self.sections_collection = self.db["penal_code_sections"]
        
        # Initialize ChromaDB with DEFAULT embeddings (no download)
        self.chroma_client = chromadb.PersistentClient(path="./law_vector_db")
        
        # Clear existing collection
        try:
            self.chroma_client.delete_collection("pakistan_penal_code")
            print("✅ Cleared existing collection")
        except:
            pass
            
        # Create collection with DEFAULT embedding function (no download)
        self.collection = self.chroma_client.get_or_create_collection(
            name="pakistan_penal_code"
            # No embedding function specified = uses default
        )
        
        print(f"✅ Initialization completed in {time.time() - start_time:.2f} seconds")
    
    def prepare_documents_simple(self):
        """Simple document preparation"""
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
            
            # Simple content - no chunking
            # if len(content) > 1500:
            #     content = content[:1500] + "..."
            
            full_content = f"Section {section_id}: {title}. {content}"
            
            documents.append(full_content)
            metadatas.append({
                "section_number": section_id,
                "section_title": title,
                "chapter": chapter,
                "source": "Pakistan Penal Code"
            })
            ids.append(str(uuid.uuid4())[:16])
        
        print(f"✅ Prepared {len(documents)} documents")
        return documents, metadatas, ids
    
    def create_vector_database_fast(self):
        """Create vector database quickly"""
        print("🔄 Creating vector database...")
        total_start = time.time()
        
        documents, metadatas, ids = self.prepare_documents_simple()
        
        if not documents:
            print("❌ No documents to process!")
            return
        
        # Add in small batches
        batch_size = 50
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
    
    # Initialize and create database
    vector_db = LocalLawVectorDB()
    vector_db.create_vector_database_fast()
    
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
                n_results=3
            )
            
            for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                print(f"   {i+1}. Section {metadata['section_number']}: {metadata['section_title']}")
                print(f"      Preview: {doc[:100]}...")
                
        except Exception as e:
            print(f"   ❌ Search failed: {e}")

if __name__ == "__main__":
    test_local_rag()