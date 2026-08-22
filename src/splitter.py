from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, min_chunk_size=100):
    """
    Split documents into chunks with filtering and error handling.

    Args:
        documents: List of LangChain Document objects
        min_chunk_size: Minimum characters per chunk

    Returns:
        List of filtered chunks

    Raises:
        ValueError: If documents is empty or splitting fails
    """

    # Validate input
    if not documents:
        raise ValueError(" No documents provided!")

    print(f"\n Processing {len(documents)} documents...")

    try:
        # Create text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )

        # Split documents
        chunks = text_splitter.split_documents(documents)

        if not chunks:
            raise ValueError(" Splitting produced no chunks!")

        print(f" Splitting created {len(chunks)} chunks")

        # Filter small/empty chunks
        original_count = len(chunks)

        chunks = [
            chunk
            for chunk in chunks
            if len(chunk.page_content.strip()) >= min_chunk_size
        ]

        removed = original_count - len(chunks)

        if not chunks:
            raise ValueError(
                f" All chunks filtered out "
                f"(min_size={min_chunk_size})"
            )

        print(f" Filtering removed {removed} small chunks")

        return chunks

    except Exception as e:
        print(f" Error: {str(e)}")
        raise


def analyze_chunks(chunks):
    """Analyze and validate chunk quality."""

    if not chunks:
        print(" No chunks to analyze")
        return False

    chunk_lengths = [
        len(chunk.page_content.strip())
        for chunk in chunks
    ]

    print("\n" + "=" * 50)
    print("CHUNK ANALYSIS")
    print("=" * 50)

    print(f"Total chunks: {len(chunks)}")
    print(f"Min size: {min(chunk_lengths)} chars")
    print(f"Max size: {max(chunk_lengths)} chars")
    print(
        f"Average: "
        f"{sum(chunk_lengths) / len(chunk_lengths):.2f} chars"
    )

    print(
        f"Median: "
        f"{sorted(chunk_lengths)[len(chunk_lengths) // 2]} chars"
    )

    # Size distribution
    print("\n SIZE DISTRIBUTION:")

    ranges = [
        (0, 100),
        (100, 500),
        (500, 1000),
        (1000, 2000),
    ]

    for start, end in ranges:

        count = sum(
            1
            for length in chunk_lengths
            if start <= length < end
        )

        if count > 0:
            percentage = (count / len(chunks)) * 100
            bar = "█" * int(percentage / 5)

            print(
                f"  {start:4d}-{end:4d}: "
                f"{count:4d} "
                f"({percentage:5.1f}%) {bar}"
            )

    # Source breakdown
    print("\n SOURCE BREAKDOWN:")

    sources = {}

    for chunk in chunks:
        source = chunk.metadata.get(
            "source",
            "unknown"
        )

        sources[source] = sources.get(source, 0) + 1

    for source, count in sources.items():

        filename = source.replace("\\", "/").split("/")[-1]

        print(f"  {filename}: {count} chunks")

    # Validation
    print("\n VALIDATION:")

    checks = {
        "Min size >= 100":
            min(chunk_lengths) >= 100,

        "Max size <= 1000":
            max(chunk_lengths) <= 1000,

        "No empty chunks":
            all(
                len(chunk.page_content.strip()) > 0
                for chunk in chunks
            ),

        "All have metadata":
            all(
                bool(chunk.metadata)
                for chunk in chunks
            ),
    }

    for check, result in checks.items():

        status = "✅" if result else "❌"

        print(f"  {status} {check}")

    return all(checks.values())


if __name__ == "__main__":

    from .loader import load_pdfs

    pdf_paths = [
        "data/raw/Machine_Learning.pdf",
        "data/raw/Deep_Learning.pdf",
    ]

    try:

        # ==============================
        # LOAD DOCUMENTS
        # ==============================

        documents = load_pdfs(pdf_paths)

        # ==============================
        # SPLIT DOCUMENTS
        # ==============================

        chunks = split_documents(
            documents,
            min_chunk_size=100
        )

        # ==============================
        # ANALYZE CHUNKS
        # ==============================

        is_valid = analyze_chunks(chunks)

        # ==============================
        # FIRST CHUNK
        # ==============================

        print("\n" + "=" * 50)
        print("FIRST CHUNK SAMPLE")
        print("=" * 50)

        print(chunks[0].page_content[:500])

        print(
            "\n Metadata:",
            chunks[0].metadata
        )

        # ==============================
        # FINAL STATUS
        # ==============================

        if is_valid:

            print(
                "\n Chunking phase completed!"
            )

            print(
                " Ready for embeddings and "
                "vector storage!"
            )

        else:

            print(
                "\n Review chunk quality "
                "before proceeding."
            )

    except Exception as e:

        print(
            f"\n Fatal error: {str(e)}"
        )

        exit(1)