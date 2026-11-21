from pathlib import Path
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

KB_DIR = Path(__file__).parent / "barangay-rag-knowledge-base"
MODEL_NAME = "gemma3:4b"


def load_knowledge_base(kb_folder: Path) -> list[Document]:
    if not kb_folder.exists():
        raise FileNotFoundError(f"Knowledge base folder not found: {kb_folder}")

    docs: list[Document] = []
    # Recursively finds .txt (and even extensionless text files) in subfolders
    for file_path in sorted(kb_folder.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".txt", ".md", ""}:
            continue

        content = file_path.read_text(encoding="utf-8").strip()
        if content:
            rel_source = str(file_path.relative_to(kb_folder))
            docs.append(
                Document(
                    page_content=f"\n{content}",
                    metadata={"source": rel_source},
                )
            )
    return docs


raw_docs = load_knowledge_base(KB_DIR)
if not raw_docs:
    raise ValueError(f"No readable text files found in {KB_DIR}")

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60)
chunks = splitter.split_documents(raw_docs)
chunk_texts = [
    f"\n{c.page_content.strip()}"
    for c in chunks
]
print(f"Loaded {len(raw_docs)} KB file(s) -> {len(chunks)} chunk(s).")

print("Loading embedding model (all-MiniLM-L6-v2)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
doc_vectors = np.array(embeddings.embed_documents(chunk_texts), dtype=np.float32)
doc_vectors_norm = doc_vectors / (np.linalg.norm(doc_vectors, axis=1, keepdims=True) + 1e-10)


def retrieve_top_k(query: str, k: int = 3) -> list[str]:
    q_vec = np.array(embeddings.embed_query(query), dtype=np.float32)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-10)
    scores = doc_vectors_norm @ q_vec
    top_indices = np.argsort(scores)[::-1][: min(k, len(chunk_texts))]
    return [chunk_texts[i] for i in top_indices]


llm = Ollama(model=MODEL_NAME, num_ctx=2048)

print("\nLocal Government Services & Ordinance Assistant ready! Type 'exit' to quit.\n")
while True:
    query = input("Resident / User question: ").strip()
    if not query:
        continue
    if query.lower() in {"exit", "quit", "q"}:
        print("Exiting assistant.")
        break

    top_chunks = retrieve_top_k(query, k=3)
    context = "\n\n---\n\n".join(top_chunks)

    prompt = f"""You are a Local Government Services and Ordinance Information Assistant.
Answer ONLY using the provided records. If the information is missing, state clearly that you do not have that information in the current knowledge base.
Include the source file name when helpful.

Records:
{context}

Question:
{query}
"""
    print("\nResponse:")
    print(llm.invoke(prompt))
    print("-" * 50)