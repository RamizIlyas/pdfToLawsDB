import chromadb
from chromadb.utils import embedding_functions
import requests
from sentence_transformers import util #, SentenceTransformer 

###Old Scoring Method using Sentence Transformers 
# - Not used as it is not strict and can give high scores to partially correct answers

# scoring_model = SentenceTransformer("all-MiniLM-L6-v2")
# Simple cosine similarity scoring function to SCore Model Output against Ground Truth
# def score_answer(predicted, ground_truth):
#     emb1 = scoring_model.encode(predicted, convert_to_tensor=True)
#     emb2 = scoring_model.encode(ground_truth, convert_to_tensor=True)
#     similarity = util.cos_sim(emb1, emb2).item()
#     return round(similarity * 100, 2)

# Adjust score based on strict judging criteria
def adjust_score(judgment_text):
    try:
        score_line = judgment_text.split("\n")[0]
        score_str = score_line.split(":")[1].strip().replace("%", "")
        score = float(score_str)
    except:
        return 50  # fallback instead of 0

    if "Faithful: No" in judgment_text:
        score -= 30

    if "Correct: No" in judgment_text:
        score -= 30
    elif "Correct: Partial" in judgment_text:
        score -= 10

    return max(score, 0)


# Enhanced evaluation with strict judging and verification using LLM
def evaluate(rag):
    print("\n📊 Running STRICT Evaluation...\n")

    total_score = 0

    for test in test_cases:
        print(f"\n❓ Question: {test['question']}")

        # Retrieve context
        docs, metas = rag.retrieve(test["question"])
        context = "\n\n".join(docs)

        # Generate answer
        predicted = rag.ask(test["question"], docs, metas)

        # Judge evaluation
        judgment = rag.judge_answer(
            test["question"],
            predicted,
            context,
            test["answer"]
        )

        # Verification check
        verification = rag.verify_answer(predicted, context)

        # Adjust score
        score = adjust_score(judgment)

        if "supported: no" in verification.lower():
            score -= 25

        score = max(score, 0)

        # Print results
        print(f"\n🤖 Answer:\n{predicted}")
        print(f"\n⚖️ Judge:\n{judgment}")
        print(f"\n🔍 Verification: {verification}")
        print(f"\n📈 Final Score: {score}")

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
    # Initialize ChromaDB client and set up collection
    def __init__(self):
        print("🚀 Initializing RAG pipeline...")

        self.chroma_client = chromadb.PersistentClient(path="./law_vector_db")

        self.collection = self.chroma_client.get_collection(
            name="pakistan_penal_code",
            embedding_function=embedding_function
        )

        # Ollama endpoint
        self.ollama_url = "http://localhost:11434/api/generate"
        self.model = "mistral"   # llama3 or mistral(Fast)
        self.judge_model = "llama3"  # For evaluation (Llama 3 is better at scoring as it is strict)

        # Warm up the model with a dummy request to reduce latency on first real query
        print("🔥 Warming up model...")
        requests.post(
            self.ollama_url,
            json={
                "model": self.model,
                "prompt": "Hello",
                "stream": False
            }
        )

    # Retrieve relevant documents from ChromaDB based on query
    def retrieve(self, query, k=3):
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )
        return results["documents"][0], results["metadatas"][0]
    
    # Build prompt for LLM generation using retrieved documents and query
    def build_prompt(self, query, docs):
        context = "\n\n".join(docs)

        prompt = f"""You are a legal assistant for Pakistan Penal Code.

                    Use ONLY the context below to answer the question.

                    Context:
                    {context}

                    Question:
                    {query}

                    Answer clearly with section references AND short explanation.
                    """
        return prompt

    # Generate answer using Mistral(Model is Mistral, Judge is Llama 3) LLM based on the built prompt
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
    
    # Judge the generated answer against the ground truth using LLM
    # , providing a score and feedback based on strict criteria

    def judge_answer(self, question, answer, context, ground_truth):
        judge_prompt = f"""You are a STRICT legal evaluator.

        Evaluate ONLY using the context.

        Question:
        {question}

        Context:
        {context}

        Model Answer:
        {answer}

        Expected Answer:
        {ground_truth}

        Rules:
        - Penalize hallucinations
        - Penalize missing details
        - Be strict

        Output EXACTLY in this format (no % sign):

        Score: <number>
        Faithful: <Yes/No>
        Correct: <Yes/Partial/No>
        Reason: <short>
        """
        response = requests.post(
            self.ollama_url,
            json={
                "model": self.judge_model,
                "prompt": judge_prompt,
                "stream": False
            }
        )

        return response.json()["response"]
    
    # Verify if the generated answer is fully supported by the retrieved context using LLM
    def verify_answer(self, answer, context):
        prompt = f"""Check if the answer is fully supported by the context.

        Answer:
        {answer}

        Context:
        {context}

        Reply ONLY:
        Supported: Yes/No
        """

        response = requests.post(
            self.ollama_url,
            json={
                "model": self.judge_model,
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"]
    
    # Main method to ask a question, retrieve context, generate answer, and print results
    def ask(self, query, docs=None, metas=None):
        if docs is None:
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