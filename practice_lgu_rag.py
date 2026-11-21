import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

MODEL_NAME = "gemma3:4b"

# 1. Sample text: Local Government Services & Ordinances (for practice)
SAMPLE_LGU_TEXT = """
[Service: Business Permit Renewal]
Business permits must be renewed annually from January 2 to January 20 at the Municipal Business Permits and Licensing Office (BPLO).
Late renewals after January 20 incur a 25% surcharge on the base tax plus 2% monthly interest.
Required documents: DTI/SEC registration, barangay clearance, lease contract or land title, and prior year's tax receipt.

[Ordinance No. 2024-08: Solid Waste Segregation & Curfew]
All households and establishments must segregate biodegradable, non-biodegradable, and recyclable waste.
Garbage collection runs every Monday, Wednesday, and Friday from 5:00 AM to 8:00 AM.
Throwing garbage on streets or waterways carries a first-offense fine of PHP 500, second offense PHP 1,500, and third offense PHP 2,500 or community service.
Youth curfew for minors under 18 without a guardian is from 10:00 PM to 4:00 AM daily, except during school-sanctioned events or emergencies.

[Service: Certificate of Indigency & Senior Citizen ID]
Residents may request a Certificate of Indigency at the Barangay Hall or Municipal Social Welfare and Development Office (MSWDO) with a valid ID.
Senior citizens aged 60 and above can apply for an OSCA/Senior Citizen ID at MSWDO using a birth certificate or valid ID and two 1x1 photos.
"""

# 2. Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
chunks = splitter.create_documents([SAMPLE_LGU_TEXT])
chunk_texts = [doc.page_content.strip() for doc in chunks]
print(f"Ingested sample text into {len(chunk_texts)} chunks.")

# 3. Embed chunks locally
print("Loading embedding model (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
doc_vectors = np.array(embeddings.embed_documents(chunk_texts), dtype=np.float32)
doc_vectors_norm = doc_vectors / (np.linalg.norm(doc_vectors, axis=1, keepdims=True) + 1e-10)


def retrieve_top_k(query: str, k: int = 2) -> list[str]:
    q_vec = np.array(embeddings.embed_query(query), dtype=np.float32)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)
    scores = doc_vectors_norm @ q_vec
    top_indices = np.argsort(scores)[::-1][:k]
    return [chunk_texts[i] for i in top_indices]


# 4. Connect to Ollama
llm = Ollama(model=MODEL_NAME, num_ctx=2048)

print("\nLGU RAG Practice Ready! Type a question (or 'quit' to stop).\n")

while True:
    user_question = input("Ask a question: ").strip()

    if not user_question:
        continue
    if user_question.lower() in {"quit", "exit", "q"}:
        print("Goodbye!")
        break

    retrieved_chunks = retrieve_top_k(user_question, k=2)
    context = "\n\n---\n\n".join(retrieved_chunks)

    # Helpful debug view so you can see what RAG found before the LLM answers
    print("\n[Retrieved Context]")
    print(context)

    prompt = f"""You are a helpful Local Government Services and Ordinance Information Assistant.
Answer the user's question using ONLY the provided context.
If the context does not contain the answer, say clearly that you do not have that information in the current records.

Context:
{context}

Question:
{user_question}
"""

    print("\n[Assistant Answer]")
    answer = llm.invoke(prompt)
    print(answer)
    print("-" * 50)