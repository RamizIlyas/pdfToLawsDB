import chromadb
from chromadb.utils import embedding_functions
import requests
from sentence_transformers import SentenceTransformer, util


scoring_model = SentenceTransformer("all-MiniLM-L6-v2")

# Simple cosine similarity scoring function to SCore Model Output against Ground Truth
def score_answer(predicted, ground_truth):
    emb1 = scoring_model.encode(predicted, convert_to_tensor=True)
    emb2 = scoring_model.encode(ground_truth, convert_to_tensor=True)

    similarity = util.cos_sim(emb1, emb2).item()

    return round(similarity * 100, 2)


# Evaluation function to test RAG pipeline on predefined test cases
def evaluate(rag):
    print("\n📊 Running Evaluation...\n")

    total_score = 0

    for test in test_cases:
        print(f"❓ Question: {test['question']}")

        predicted = rag.ask(test["question"])
        score = score_answer(predicted, test["answer"])

        print(f"🤖 Model Answer: {predicted}")
        print(f"✅ Expected: {test['answer']}")
        print(f"📈 Score: {score}%\n")

        total_score += score

    avg = total_score / len(test_cases)
    print(f"\n🏁 Final Accuracy: {avg:.2f}%")

# Use SAME embedding function as your DB
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-base-en-v1.5",
    normalize_embeddings=True
)

test_cases = [
    {
        "question": "What are punishments under Section 53?",
        "answer": "Qisas, Diyat, Arsh, Daman, Ta'zir, Death, Imprisonment for life, Rigorous imprisonment, Simple imprisonment, Forfeiture of property, Fine"
    },
    {
        "question": "Can death sentence be commuted without consent?",
        "answer": "Yes, but not in qatl cases without consent of heirs"
    },
    {
        "question": "How is life imprisonment calculated?",
        "answer": "Life imprisonment is considered as 25 years"
    },
    {
        "question": "What happens if fine is partially paid?",
        "answer": "Imprisonment may be reduced proportionally"
    }
]

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

                    Answer clearly with section references AND short explanation.
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
    evaluate(rag)
    while True:
        query = input("\n💬 Ask a legal question (or 'exit'): ")
        if query.lower() == "exit":
            break

        answer = rag.ask(query)
        print("\n🤖 Answer:\n", answer)