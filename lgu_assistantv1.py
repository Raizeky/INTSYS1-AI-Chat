
from pathlib import Path

import numpy as np

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
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
                        "filename": file_path.name
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
# 3. SPLIT DOCUMENTS INTO SERVICE-BASED CHUNKS
# ============================================================

print("\nSplitting documents into chunks...")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = []

for doc in raw_docs:

    source = doc.metadata.get("source", "")

    # Special handling for Katarungang Pambarangay
    if "09_Katarungang_Pambarangay" in source:

        import re

        # Identify the main service headings
        pattern = r"(?=^[ \t]*[A-D]\.\s+)"

        sections = re.split(
            pattern,
            doc.page_content,
            flags=re.MULTILINE
        )

        for section in sections:

            section = section.strip()

            if not section:
                continue

            # Preserve the original document metadata
            section_doc = Document(
                page_content=section,
                metadata={
                    **doc.metadata,
                    "service_section": section.split("\n")[0].strip()
                }
            )

            # Split only if an individual service is too long
            section_chunks = splitter.split_documents(
                [section_doc]
            )

            chunks.extend(section_chunks)

    else:

        # Keep existing chunking for all other documents
        chunks.extend(
            splitter.split_documents([doc])
        )

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
# 5. Remove or prevent duplicate retrieval
# ============================================================

def remove_duplicate_chunks(docs):
    unique_docs = []
    seen = set()

    for doc in docs:
        normalized = " ".join(
            doc.page_content.lower().split()
        )

        if normalized not in seen:
            seen.add(normalized)
            unique_docs.append(doc)

    return unique_docs

# ============================================================
# 5. RETRIEVE TOP-K RELEVANT DOCUMENTS
# ============================================================

def retrieve_top_k(
    query: str,
    k: int = TOP_K
) -> list[Document]:

    if not query.strip():
        return []

    q_vec = np.array(
        embeddings.embed_query(query),
        dtype=np.float32
    )

    q_vec = q_vec / (
        np.linalg.norm(q_vec) + 1e-10
    )

    scores = doc_vectors_norm @ q_vec

    # Retrieve more candidates before filtering
    candidate_count = min(k * 3, len(chunks))

    top_indices = np.argsort(scores)[::-1][
        :candidate_count
    ]

    retrieved_docs = []

    for index in top_indices:

        doc = chunks[index]

        retrieved_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "similarity_score": float(scores[index])
                }
            )
        )

    # Remove exact duplicate chunks
    retrieved_docs = remove_duplicate_chunks(
        retrieved_docs
    )

    # Keep only the final TOP_K results
    return retrieved_docs[:k]


# ============================================================
# 6. INITIALIZE OLLAMA LLM
# ============================================================

print("\nConnecting to Ollama...")

#llm = ChatOllama(
#    model=LLM_MODEL,
#    temperature=0,
#    num_ctx=2048
#)
llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
    num_ctx=4096,
    num_predict=1024
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
[Retrieved Chunk {index}]
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

Your responsibility is to answer residents' questions
using ONLY the relevant information found in the
provided Barangay Knowledge Base records.

STRICT ANSWERING RULES:

1. USE ONLY RELEVANT INFORMATION

   - Only use records that are directly relevant to the
     resident's question.
   - Do not use unrelated records just because they
     appear in the retrieved context.
   - Never invent information or use unsupported facts.

2. HANDLE MISSING INFORMATION CORRECTLY

   - If the requested information is completely available,
     answer the question directly.
   - If some requested information is available but
     other information is missing, provide the available
     information and identify only the missing details.
   - If the requested information is not available in
     the relevant records, state:

     "I could not find this information in the current
     barangay knowledge base."

   - NEVER append the missing-information statement
     after successfully answering the question.
   - Do not say information is missing if it is already
     explicitly stated in the records.

3. DO NOT MAKE ASSUMPTIONS

   Never invent:
   - Requirements
   - Fees or costs
   - Processing times
   - Office schedules
   - Procedures
   - Government policies

4. ANSWER THE ACTUAL QUESTION

   - If the resident asks about requirements, focus on
     the requirements.
   - If the resident asks about fees, focus on fees.
   - If the resident asks about procedures, explain
     the applicable steps.
   - Do not unnecessarily include unrelated sections.

5. ORGANIZE YOUR ANSWERS

   When appropriate, organize answers into:

   - Requirements
   - Procedure
   - Fees or Cost
   - Important Information

   Only include sections relevant to the question and
   supported by the knowledge base.

6. SOURCE ATTRIBUTION

   - Mention the source filename when possible.
   - Do not invent record numbers or source references.
   - Do not attribute information to a source that
     does not contain it.

7. MULTIPLE RECORDS

   - Combine relevant information from multiple records
     when necessary.
   - Do not combine different services just because
     their names or procedures are similar.
   - Do not mix up their requirements, fees, or procedures.

8. OUT-OF-SCOPE QUESTIONS

   If the question is unrelated to barangay services
   or ordinances, politely explain your role as a
   Barangay Services and Ordinance Information Assistant.

9. FEES AND COSTS

   - If the record explicitly states that a service is
     free, clearly state that it is free.
   - If a specific fee is provided, state the exact amount.
   - If payment is required but the amount is unspecified,
     explain that the exact amount is not available.
   - If no fee information is provided, do not assume
     that the service is free.
   - Only use the missing-information response when
     the resident specifically asks about fees.

10. HEALTH-RELATED QUESTIONS

   - Only provide health service information explicitly
     stated in the knowledge base.
   - Do not diagnose medical conditions.
   - Do not recommend medications, dosages, or treatments
     that are not specified in the records.
   - For medical concerns requiring professional
     evaluation, advise the resident to consult a
     qualified healthcare professional.

11. RESPONSE QUALITY

   - Use clear, simple, and understandable language.
   - Avoid unnecessary repetition.
   - Do not add information that the resident did not ask for.
   - Do not contradict your own answer.
   - Be concise but complete.

12. SERVICE IDENTITY AND RECORD SEPARATION

    - Each knowledge base source represents a specific
      barangay service or document.

    - Always identify the service being asked about
      before generating an answer.

    - Never substitute information from one service
      with information from another service.

    - Barangay Certificate, Barangay Clearance, and
      Barangay Business Clearance are separate services.

    - Never combine their fees, requirements, or procedures.

    - If the resident asks about Barangay Business
      Clearance, prioritize information from the
      Barangay Business Clearance source.

    - Do not use the regular Barangay Clearance fee
      when answering questions about Barangay
      Business Clearance.

13. REQUIREMENT AND PROCEDURE APPLICABILITY

    - Pay attention to conditions such as:
      NEW APPLICANTS, OLD APPLICANTS, RENEWAL,
      SENIOR CITIZENS, SOLO PARENTS, and RESIDENTS.

    - Only include requirements that apply to the
      specific applicant type mentioned in the question.

    - Do not combine requirements for new applicants
      with requirements for renewal applicants.

    - Do not present conditional requirements as
      universal requirements.

    - If the knowledge base provides separate
      requirements for different applicant types,
      clearly separate them.

    - If the resident asks about a new application,
      prioritize requirements explicitly listed
      for new applicants.

    - If the requirements are incomplete, explain
      which information is available and which
      information is not specified.

14. STRICT SERVICE SEPARATION:

    Never combine different barangay services or procedures
    just because they belong to the same knowledge base document.

    For example:
    - Filing a Barangay Complaint is different from
    Certification to File Action (CFA).
    - Certification to File Action is different from
    Certification to Bar Action (CBA).
    - Barangay Certificate is different from
    Barangay Clearance.

    Only include procedures, requirements, and fees that
    directly belong to the service requested by the user.

    Do not interpret related services as alternative
    steps or options unless the knowledge base explicitly
    states that they are alternatives.

15. QUESTION-SPECIFIC ANSWERING:

    Identify the exact stage of the service being asked.

    If the user asks what happens AFTER issuance,
    only provide the procedures and information following
    the issuance.

    Do not include application requirements or
    pre-issuance procedures unless requested.

    If the user asks who can issue a document,
    answer only the authorized issuing official and
    relevant conditions.

16. EXACT SERVICE NAME MATCHING:

    When answering questions about fees, requirements,
    or procedures, identify the exact service name first.

    Do not substitute similar service names.

    Examples:

    CFA = Certification to File Action
    CBA = Certification to Bar Action
    CBC = Certificate to Bar Counter-Claim

    Never use the requirements, procedures, or fees
    of one service to answer a question about another.

    If the exact service cannot be identified from
    the retrieved knowledge, state that the information
    could not be found.

      
KNOWLEDGE BASE RECORDS:

{context}

RESIDENT QUESTION:

{query}

ANSWER:
"""

    response = llm.invoke(prompt)

    return response.content.strip()


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

            # ------------------------------------------------
            # RETRIEVE DOCUMENTS
            # ------------------------------------------------

            print("\nSearching knowledge base...")

            top_chunks = retrieve_top_k(
                query,
                k=TOP_K
            )

            top_chunks = remove_duplicate_chunks(top_chunks)

            if not top_chunks:
                print("No relevant records found.")
                continue

            # ------------------------------------------------
            # DISPLAY RETRIEVED SOURCES AND SCORES
            # ------------------------------------------------

            print("\nRetrieved sources:")

            for index, doc in enumerate(top_chunks, start=1):

                print(f"\n[Retrieved Chunk {index}]")

                print(
                    f"Source: "
                    f"{doc.metadata.get('source', 'Unknown')}"
                )

                print(
                    f"Similarity: "
                    f"{doc.metadata.get('similarity_score', 0):.4f}"
                )

                print("Content:")
                print(doc.page_content)

                print("-" * 50)

            # ------------------------------------------------
            # GENERATE ANSWER
            # ------------------------------------------------

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
