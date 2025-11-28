import chromadb
import json
from datetime import datetime
import numpy as np

def export_to_json():
    """Export ChromaDB to JSON for analysis in other tools"""
    client = chromadb.PersistentClient(path="./law_vector_db")
    collection = client.get_collection("pakistan_penal_code")
    
    # Get all data
    all_data = collection.get(include=['documents', 'metadatas', 'embeddings'])
    
    # Structure data for export
    export_data = {
        "metadata": {
            "total_documents": len(all_data['ids']),
            "collection_name": "pakistan_penal_code",
            "export_date": datetime.now().isoformat()
        },
        "documents": []
    }
    
    # Fix the embedding check
    has_embeddings = all_data.get('embeddings') is not None
    
    for i, (doc_id, document, metadata) in enumerate(zip(all_data['ids'], all_data['documents'], all_data['metadatas'])):
        export_data["documents"].append({
            "id": doc_id,
            "section_number": metadata.get('section_number'),
            "section_title": metadata.get('section_title'),
            "chapter": metadata.get('chapter'),
            "content": document,
            "content_length": len(document),
            "has_embedding": has_embeddings and i < len(all_data['embeddings'])
        })
    
    # Save to JSON file
    with open("law_database_export.json", "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print("✅ JSON export created: law_database_export.json")
    print(f"📊 Exported {len(export_data['documents'])} documents")

# Alternative simpler version without embeddings
def export_to_json_simple():
    """Simplified JSON export without embedding checks"""
    client = chromadb.PersistentClient(path="./law_vector_db")
    collection = client.get_collection("pakistan_penal_code")
    
    # Get only documents and metadata (no embeddings)
    all_data = collection.get(include=['documents', 'metadatas'])
    
    export_data = {
        "metadata": {
            "total_documents": len(all_data['ids']),
            "collection_name": "pakistan_penal_code", 
            "export_date": datetime.now().isoformat()
        },
        "documents": []
    }
    
    for doc_id, document, metadata in zip(all_data['ids'], all_data['documents'], all_data['metadatas']):
        export_data["documents"].append({
            "id": doc_id,
            "section_number": metadata.get('section_number'),
            "section_title": metadata.get('section_title'),
            "chapter": metadata.get('chapter'),
            "content": document,
            "content_length": len(document)
        })
    
    with open("law_database_simple.json", "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    print("✅ Simple JSON export created: law_database_simple.json")
    print(f"📊 Exported {len(export_data['documents'])} documents")

# Quick viewer function
def quick_view_database():
    """Quick view of what's in the database"""
    client = chromadb.PersistentClient(path="./law_vector_db")
    collection = client.get_collection("pakistan_penal_code")
    
    # Get a sample of documents
    all_data = collection.get(include=['documents', 'metadatas'])
    
    print("🔍 QUICK DATABASE VIEW")
    print("=" * 60)
    print(f"Total documents: {len(all_data['ids'])}")
    print("\nSample documents:")
    print("-" * 60)
    
    # Show first 5 documents as sample
    for i in range(min(5, len(all_data['ids']))):
        print(f"\n📄 Document {i+1}:")
        print(f"   ID: {all_data['ids'][i]}")
        print(f"   Section: {all_data['metadatas'][i].get('section_number', 'N/A')}")
        print(f"   Title: {all_data['metadatas'][i].get('section_title', 'N/A')}")
        print(f"   Content preview: {all_data['documents'][i][:200]}...")
        print("-" * 40)

if __name__ == "__main__":
    # Run the simple version first to avoid errors
    export_to_json_simple()
    quick_view_database()