import re
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="RAG Chunking Playground",
    page_icon="🧩",
    layout="wide"
)

st.title("🧩 RAG Chunking Playground")
st.caption("Explore 5 document chunking strategies interactively.")

SAMPLE_TEXT = """# Employee Leave Policy

Employees are eligible for 24 paid leaves per year.
Employees must submit leave requests at least three days in advance.
Unused paid leaves can be carried forward up to a maximum of 10 days.

# Salary and Benefits

Employees receive their salaries on the last working day of each month.
The company provides health insurance to all full-time employees.
Performance bonuses depend on individual and company performance.

# Company Holidays

The company observes 12 public holidays every year.
Employees can check the annual holiday calendar on the company portal.
Holiday schedules may vary depending on the employee's location.
"""


# --------------------------------------------------
# 1. FIXED-SIZE CHUNKING
# --------------------------------------------------

def fixed_size_chunking(text, chunk_size, overlap):
    text = text.strip()
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


# --------------------------------------------------
# 2. RECURSIVE CHUNKING
# --------------------------------------------------

def recursive_chunking(text, chunk_size):
    separators = ["\n\n", "\n", ". ", " "]

    def split_recursive(content):
        content = content.strip()

        if not content:
            return []

        if len(content) <= chunk_size:
            return [content]

        # Find the first separator that divides the text.
        for separator in separators:
            parts = content.split(separator)

            if len(parts) > 1:
                break
        else:
            # Fallback for text without separators.
            return [
                content[i:i + chunk_size]
                for i in range(0, len(content), chunk_size)
            ]

        chunks = []
        current = ""

        for part in parts:
            part = part.strip()

            if not part:
                continue

            candidate = f"{current}{separator if current else ''}{part}"

            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)

                current = ""

                if len(part) > chunk_size:
                    chunks.extend(split_recursive(part))
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    return split_recursive(text)


# --------------------------------------------------
# 3. SENTENCE-BASED CHUNKING
# --------------------------------------------------

def sentence_chunking(text, sentences_per_chunk):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    return [
        " ".join(sentences[i:i + sentences_per_chunk])
        for i in range(0, len(sentences), sentences_per_chunk)
    ]


# --------------------------------------------------
# 4. SEMANTIC CHUNKING
# --------------------------------------------------

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def semantic_chunking(text, threshold):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        return sentences

    model = load_embedding_model()
    embeddings = model.encode(sentences)

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        similarity = cosine_similarity(
            [embeddings[i - 1]],
            [embeddings[i]]
        )[0][0]

        if similarity >= threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# --------------------------------------------------
# 5. DOCUMENT-STRUCTURE-BASED CHUNKING
# --------------------------------------------------

def structure_based_chunking(text, max_size):
    """
    Split Markdown documents using headings.
    If a section is too large, split it recursively.
    """
    sections = re.split(
        r"(?m)(?=^#{1,6}\s+)",
        text.strip()
    )

    chunks = []

    for section in sections:
        section = section.strip()

        if not section:
            continue

        if len(section) <= max_size:
            chunks.append(section)
        else:
            chunks.extend(recursive_chunking(section, max_size))

    return chunks


# --------------------------------------------------
# USER INTERFACE
# --------------------------------------------------

st.subheader("📄 Step 1: Enter Your Document")

document = st.text_area(
    "Paste your document",
    value=SAMPLE_TEXT,
    height=260
)

strategy = st.selectbox(
    "Step 2: Choose a Chunking Strategy",
    [
        "Fixed-Size Chunking",
        "Recursive Chunking",
        "Sentence-Based Chunking",
        "Semantic Chunking",
        "Document-Structure-Based Chunking"
    ]
)

chunk_size = 300
overlap = 30
sentences_per_chunk = 3
threshold = 0.45

if strategy == "Fixed-Size Chunking":
    chunk_size = st.slider(
        "Chunk size (characters)",
        100, 1000, 300, 50
    )

    overlap = st.slider(
        "Overlap (characters)",
        0, 100, 30, 10
    )

    # Prevent overlap from being equal to chunk size.
    overlap = min(overlap, chunk_size - 1)

elif strategy == "Recursive Chunking":
    chunk_size = st.slider(
        "Maximum chunk size (characters)",
        100, 1000, 300, 50
    )

elif strategy == "Sentence-Based Chunking":
    sentences_per_chunk = st.slider(
        "Sentences per chunk",
        1, 10, 3
    )

elif strategy == "Semantic Chunking":
    threshold = st.slider(
        "Similarity threshold",
        0.10, 0.90, 0.45, 0.05
    )

elif strategy == "Document-Structure-Based Chunking":
    chunk_size = st.slider(
        "Maximum section size (characters)",
        100, 1500, 500, 50
    )

st.info({
    "Fixed-Size Chunking":
        "Splits text into fixed-length blocks.",
    "Recursive Chunking":
        "Tries natural text separators before smaller splits.",
    "Sentence-Based Chunking":
        "Groups a fixed number of sentences together.",
    "Semantic Chunking":
        "Groups neighboring sentences based on embedding similarity.",
    "Document-Structure-Based Chunking":
        "Uses Markdown headings to identify document sections."
}[strategy])


# --------------------------------------------------
# RUN AND DISPLAY RESULTS
# --------------------------------------------------

if st.button("✂️ Generate Chunks", type="primary",
             use_container_width=True):

    if not document.strip():
        st.warning("Please enter a document.")

    else:
        try:
            with st.spinner("Processing document..."):

                if strategy == "Fixed-Size Chunking":
                    chunks = fixed_size_chunking(
                        document, chunk_size, overlap
                    )

                elif strategy == "Recursive Chunking":
                    chunks = recursive_chunking(
                        document, chunk_size
                    )

                elif strategy == "Sentence-Based Chunking":
                    chunks = sentence_chunking(
                        document, sentences_per_chunk
                    )

                elif strategy == "Semantic Chunking":
                    chunks = semantic_chunking(
                        document, threshold
                    )

                else:
                    chunks = structure_based_chunking(
                        document, chunk_size
                    )

            st.divider()
            st.subheader("📊 Chunking Results")

            col1, col2, col3 = st.columns(3)

            col1.metric("Total Chunks", len(chunks))
            col2.metric("Document Characters", len(document))

            average_size = (
                sum(len(c) for c in chunks) // len(chunks)
                if chunks else 0
            )

            col3.metric("Average Chunk Size", f"{average_size} chars")

            for index, chunk in enumerate(chunks, start=1):
                with st.container(border=True):
                    st.markdown(f"### Chunk {index}")
                    st.caption(f"{len(chunk)} characters")
                    st.write(chunk)

            st.download_button(
                "⬇️ Download Chunks",
                data="\n\n--- CHUNK BREAK ---\n\n".join(chunks),
                file_name="rag_chunks.txt",
                mime="text/plain",
                use_container_width=True
            )

        except Exception as error:
            st.error(f"Error: {error}")


# --------------------------------------------------
# STRATEGY COMPARISON
# --------------------------------------------------

st.divider()
st.subheader("💡 Compare the Five Strategies")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1. Fixed-Size")
    st.write("Splits text at a fixed character limit.")
    st.caption("Simple and fast.")

with col2:
    st.markdown("### 2. Recursive")
    st.write("Tries natural boundaries before splitting further.")
    st.caption("A useful general-purpose starting point.")

with col3:
    st.markdown("### 3. Sentence-Based")
    st.write("Groups a specified number of sentences.")
    st.caption("Useful for sentence-oriented content.")

col4, col5 = st.columns(2)

with col4:
    st.markdown("### 4. Semantic")
    st.write("Uses sentence embeddings to group neighboring sentences.")
    st.caption("Useful when meaning-based boundaries matter.")

with col5:
    st.markdown("### 5. Structure-Based")
    st.write("Splits using headings and document sections.")
    st.caption("Useful for structured documents.")