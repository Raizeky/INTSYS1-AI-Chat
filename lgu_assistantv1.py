
from pathlib import Path

import numpy as np

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama


# ============================================================
# CONFIGURATION
# ============================================================

KB_DIR = Path(__file__).parent / "barangay-rag-knowledge-base"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemma3:4b"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 5


# ============================================================
# 1. LOAD KNOWLEDGE BASE
# ============================================================

def load_knowledge_base(kb_folder: Path) -> list[Document]:

    if not kb_folder.exists():
        raise FileNotFoundError(
            f"Knowledge base folder not found: {kb_folder}"
        )

    docs = []

    supported_extensions = {".txt", ".md", ""}

    for file_path in sorted(kb_folder.rglob("*")):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in supported_extensions:
            continue

        try:
            content = file_path.read_text(
                encoding="utf-8"
            ).strip()

            if not content:
                print(f"Skipping empty file: {file_path.name}")
                continue

            rel_source = str(
                file_path.relative_to(kb_folder)
            )

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": rel_source,
                        "filename": file_path.name,
                        "fee": "500.00",
                        # FUTURE IMPLEMENTATION:
                        # Metadata-aware retrieval and document-level organization

                        # "category": "barangay_ordinances",
                        # "service": "waste_management",
                        # "document_type": "ordinance"
                    }
                )
            )

            print(f"Loaded: {rel_source}")

        except (UnicodeDecodeError, OSError) as e:
            print(f"Could not read {file_path}: {e}")

    return docs


# ============================================================
# 2. LOAD AND VALIDATE DOCUMENTS
# ============================================================

print("\nLoading Barangay Knowledge Base...")

raw_docs = load_knowledge_base(KB_DIR)

if not raw_docs:
    raise ValueError(
        f"No readable text files found in {KB_DIR}"
    )

print(f"\nTotal documents loaded: {len(raw_docs)}")


# ============================================================
# 3. SPLIT DOCUMENTS INTO CHUNKS
# ============================================================

print("\nSplitting documents into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(raw_docs)

if not chunks:
    raise ValueError(
        "No document chunks were generated."
    )

print(f"Total chunks generated: {len(chunks)}")


# ============================================================
# 4. LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

chunk_texts = [
    chunk.page_content
    for chunk in chunks
]

print("Generating document embeddings...")

doc_vectors = np.array(
    embeddings.embed_documents(chunk_texts),
    dtype=np.float32
)

# Normalize vectors for cosine similarity
doc_vectors_norm = doc_vectors / (
    np.linalg.norm(
        doc_vectors,
        axis=1,
        keepdims=True
    ) + 1e-10
)

print("Document embeddings generated successfully.")


# ============================================================
# 5. RETRIEVE TOP-K RELEVANT DOCUMENTS
# ============================================================

def retrieve_top_k(
    query: str,
    k: int = TOP_K
) -> list[Document]:

    if not query.strip():
        return []

    # Generate query embedding
    q_vec = np.array(
        embeddings.embed_query(query),
        dtype=np.float32
    )

    # Normalize query vector
    q_vec = q_vec / (
        np.linalg.norm(q_vec) + 1e-10
    )

    # Calculate cosine similarity
    scores = doc_vectors_norm @ q_vec

    # Get highest scoring chunks
    top_indices = np.argsort(scores)[::-1][
        :min(k, len(chunks))
    ]

    return [chunks[i] for i in top_indices]


# ============================================================
# 6. INITIALIZE OLLAMA LLM
# ============================================================

print("\nConnecting to Ollama...")

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
    num_ctx=2048
)

print("Ollama connected successfully.")


# ============================================================
# 7. BUILD CONTEXT WITH SOURCE INFORMATION
# ============================================================

def build_context(
    retrieved_docs: list[Document]
) -> str:

    context_parts = []

    for index, doc in enumerate(retrieved_docs, start=1):

        source = doc.metadata.get(
            "source",
            "Unknown source"
        )

        content = doc.page_content.strip()

        context_parts.append(
            f"""
[Record {index}]
Source: {source}

Content:
{content}
"""
        )

    return "\n\n---\n\n".join(context_parts)


# ============================================================
# 8. GENERATE ANSWER
# ============================================================

def generate_answer(
    query: str,
    retrieved_docs: list[Document]
) -> str:

    context = build_context(retrieved_docs)

    prompt = f"""
You are a Barangay Services and Ordinance Information Assistant.

Your task is to answer questions from residents using the
provided Barangay Knowledge Base records.

IMPORTANT INSTRUCTIONS:

1. Answer ONLY using the provided knowledge base records.

2. Never invent requirements, fees, processing times,
   procedures, office schedules, or government policies.

3. If the requested information is missing, clearly state:
   "I could not find this information in the current
   barangay knowledge base."

4. Provide clear, concise, and understandable answers.

5. When explaining a service, organize the answer into
   useful sections such as:
   - Requirements
   - Procedure
   - Fees or Cost

   Only include sections supported by the records.

6. Mention the source filename whenever possible.

7. If multiple records contain relevant information,
   combine them without adding unsupported details.

8. If the question is unrelated to barangay services or
   ordinances, politely explain your role.

9. When handling fees or costs:
   - If the record states the service is free or has no fee,
     state clearly that there is no fee (or it is free) as
     indicated in the record.
   - If the record says the fee varies, depends on conditions,
     or is subject to assessment/computation by the office,
     state that no fixed amount is specified in the record and
     explain what factors or assessment apply based only on
     the text.
   - Only use the "missing information" fallback if the
     record does not mention fees or payment at all and the
     user specifically asked about the cost.

10. For health-related questions:
    - Only provide health service information explicitly
      stated in the knowledge base.
    - Do not diagnose medical conditions.
    - Do not recommend medications, dosages, or treatments
      that are not specified in the records.
    - For medical concerns requiring professional
      evaluation, advise the resident to consult a
      qualified healthcare professional.

KNOWLEDGE BASE RECORDS:

{context}

RESIDENT QUESTION:

{query}

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content


# ============================================================
# 9. INTERACTIVE RAG CHAT
# ============================================================

def main():

    print("\n" + "=" * 60)
    print(" LOCAL GOVERNMENT SERVICES & ORDINANCE ASSISTANT")
    print("=" * 60)

    print(f"Knowledge Base Documents : {len(raw_docs)}")
    print(f"Generated Chunks         : {len(chunks)}")
    print(f"Embedding Model          : {EMBEDDING_MODEL}")
    print(f"LLM Model                : {LLM_MODEL}")

    print("\nAssistant ready!")
    print("Type 'exit', 'quit', or 'q' to stop.\n")

    while True:

        try:
            query = input("Resident / User question: ").strip()

            if not query:
                continue

            if query.lower() in {"exit", "quit", "q"}:

                print("\nExiting assistant.")
                break

            # Retrieve relevant documents
            print("\nSearching knowledge base...")

            top_chunks = retrieve_top_k(
                query,
                k=TOP_K
            )

            if not top_chunks:
                print("No relevant records found.")
                continue

            # Display retrieved sources
            print("\nRetrieved sources:")

            for doc in top_chunks:
                print(
                    f"- {doc.metadata.get('source', 'Unknown')}"
                )

            # Generate response
            print("\nGenerating response...")

            answer = generate_answer(
                query,
                top_chunks
            )

            print("\n" + "=" * 60)
            print("ASSISTANT RESPONSE")
            print("=" * 60)

            print(answer)

            print("\n" + "-" * 60)

        except KeyboardInterrupt:
            print("\n\nAssistant stopped.")
            break

        except Exception as e:
            print(f"\nAn error occurred: {e}")


# ============================================================
# 10. RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()
