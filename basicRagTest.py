import chromadb
from pymongo import MongoClient

def basic_rag_test():
    """Test basic RAG functionality"""
    print("🧪 BASIC RAG FUNCTIONALITY TEST")
    print("=" * 50)
    
    # Connect to your vector database
    client = chromadb.PersistentClient(path="./law_vector_db")
    collection = client.get_collection("pakistan_penal_code")
    
    # Connect to MongoDB for reference
    mongo_client = MongoClient("mongodb://localhost:27017/")
    db = mongo_client["pakistan_law_db"]
    sections_coll = db["penal_code_sections"]
    
    # Test 1: Database Count
    print("\n1. 📊 DATABASE STATISTICS:")
    doc_count = collection.count()
    print(f"   Documents in vector database: {doc_count}")
    
    # Test 2: Basic Searches
    print("\n2. 🔍 BASIC SEARCH TESTS:")
    
    test_queries = [
        "murder punishment",
        "theft robbery", 
        "assault hurt injury",
        "fraud cheating",
        "property damage",
        "criminal force",
        "wrongful restraint",
        "false evidence"
    ]
    
    for query in test_queries:
        print(f"\n   Searching: '{query}'")
        try:
            results = collection.query(
                query_texts=[query],
                n_results=3
            )
            
            if results['documents']:
                for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    print(f"      {i+1}. Section {metadata['section_number']}: {metadata['section_title']}")
                    # print(f"         Preview: {doc[:1500]}...")
                    print(f"         Preview: {doc}")
            else:
                print("      No results found")
                
        except Exception as e:
            print(f"      ❌ Error: {e}")
    
    # Test 3: Get specific section
    print("\n3. 📖 SPECIFIC SECTION LOOKUP:")
    important_sections = ["302", "379", "420", "497", "34"]
    
    for section_num in important_sections:
        try:
            section_results = collection.get(
                where={"section_number": section_num},
                include=["documents", "metadatas"]
            )
            
            if section_results['documents']:
                print(f"   ✅ Section {section_num} found: {section_results['metadatas'][0]['section_title']}")
            else:
                print(f"   ❌ Section {section_num} not found")
                
        except Exception as e:
            print(f"   ❌ Error retrieving section {section_num}: {e}")
     
    mongo_client.close()
    print("\n🎉 BASIC TESTING COMPLETED!")

if __name__ == "__main__":
    basic_rag_test()