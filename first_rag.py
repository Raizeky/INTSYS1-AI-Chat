from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np

PDF_FILE = "Ollama Lab Guide.pdf"
MODEL_NAME = "llama3:8b"  # Uses less VRAM than llama3:8b

pdf_path = Path(__file__).parent / PDF_FILE
if not pdf_path.exists():
  raise FileNotFoundError(f"PDF not found: {pdf_path}")

# 1. Load and chunk the PDF
loader = PyPDFLoader(str(pdf_path))
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
chunk_texts = [doc.page_content for doc in chunks]
print(f"Loaded {len(documents)} pages -> created {len(chunk_texts)} chunks.")

# 2. Embed chunks locally with all-MiniLM-L6-v2
print("Embedding chunks with all-MiniLM-L6-v2...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
doc_vectors = np.array(
    embeddings.embed_documents(chunk_texts), dtype=np.float32
)

doc_norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True) + 1e-10
doc_vectors_norm = doc_vectors / doc_norms


def retrieve_top_k(query: str, k: int = 3) -> list[str]:
  q_vec = np.array(embeddings.embed_query(query), dtype=np.float32)
  q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)
  scores = doc_vectors_norm @ q_vec
  top_indices = np.argsort(scores)[::-1][:k]
  return [chunk_texts[i] for i in top_indices]


# 3. Retrieve context and generate answer with Ollama
llm = Ollama(model=MODEL_NAME, num_ctx=2048)
question = "What happens in Lab 2.1?"

top_chunks = retrieve_top_k(question, k=3)
context = "\n\n---\n\n".join(top_chunks)

prompt = f"""Answer the question using ONLY the context below. If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}
"""

print(f"\nAsking {MODEL_NAME}: {question}\n")
answer = llm.invoke(prompt)
print("Answer:\n", answer)