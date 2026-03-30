import chromadb
from chromadb.utils import embedding_functions
import requests

# Use SAME embedding function as your DB
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-base-en-v1.5",
    normalize_embeddings=True
)

class LocalRAG:
    def __init__(self):
        print("🚀 Initializing RAG pipeline...")

        self.chroma_client = chromadb.PersistentClient(path="./law_vector_db")

        self.collection = self.chroma_client.get_collection(
            name="pakistan_penal_code",
            embedding_function=embedding_function
        )

        # Ollama endpoint
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "llama3"   # or mistral

    def retrieve(self, query, k=5):
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return results["documents"][0], results["metadatas"][0]

    def build_prompt(self, query, docs):
        context = "\n\n".join(docs)

        prompt = f"""
                    You are a legal assistant for Pakistan Penal Code.

                    Use ONLY the context below to answer the question.

                    Context:
                    {context}

                    Question:
                    {query}

                    Answer clearly with section references when possible.
                    """
        return prompt

    def generate(self, prompt):
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"]

    def ask(self, query):
        docs, metas = self.retrieve(query)

        print("\n📚 Retrieved Context:")
        for m in metas[:3]:
            print(f"   Section {m['section_number']} - {m['section_title']}")

        prompt = self.build_prompt(query, docs)
        answer = self.generate(prompt)

        return answer


if __name__ == "__main__":
    rag = LocalRAG()

    while True:
        query = input("\n💬 Ask a legal question (or 'exit'): ")
        if query.lower() == "exit":
            break

        answer = rag.ask(query)
        print("\n🤖 Answer:\n", answer)