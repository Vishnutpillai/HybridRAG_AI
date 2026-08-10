from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, min_chunk_size=100):
    """
    Split documents into chunks with filtering and error handling.
    
    Args:
        documents: List of LangChain Document objects
        min_chunk_size: Minimum characters per chunk (default: 100)
        
    Returns:
        List of filtered chunks
        
    Raises:
        ValueError: If documents is empty or splitting fails
    """
    # ✅ Validate input
    if not documents:
        raise ValueError("❌ No documents provided!")
    
    print(f"\n📚 Processing {len(documents)} documents...")
    
    try:
        # Create splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Split documents
        chunks = text_splitter.split_documents(documents)
        
        if not chunks:
            raise ValueError("❌ Splitting produced no chunks!")
        
        print(f"✅ Splitting created {len(chunks)} chunks")
        
        # Filter by minimum size
        original_count = len(chunks)
        chunks = [c for c in chunks if len(c.page_content) >= min_chunk_size]
        removed = original_count - len(chunks)
        
        if not chunks:
            raise ValueError(f"❌ All chunks filtered out (min_size={min_chunk_size})")
        
        print(f"✅ Filtering removed {removed} small chunks")
        
        return chunks
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise


def analyze_chunks(chunks):
    """Analyze and validate chunk quality."""
    
    if not chunks:
        print("❌ No chunks to analyze")
        return False
    
    chunk_lengths = [len(c.page_content) for c in chunks]
    
    print("\n" + "="*50)
    print("CHUNK ANALYSIS")
    print("="*50)
    print(f"Total chunks: {len(chunks)}")
    print(f"Min size: {min(chunk_lengths)} chars")
    print(f"Max size: {max(chunk_lengths)} chars")
    print(f"Average: {sum(chunk_lengths)/len(chunk_lengths):.2f} chars")
    print(f"Median: {sorted(chunk_lengths)[len(chunk_lengths)//2]} chars")
    
    # Distribution
    print("\n📊 SIZE DISTRIBUTION:")
    ranges = [(0, 100), (100, 500), (500, 1000), (1000, 2000)]
    for start, end in ranges:
        count = sum(1 for l in chunk_lengths if start <= l < end)
        if count > 0:
            pct = (count/len(chunks))*100
            bar = "█" * int(pct/5)
            print(f"  {start:4d}-{end:4d}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Metadata
    print("\n📋 SOURCE BREAKDOWN:")
    sources = {}
    for chunk in chunks:
        source = chunk.metadata.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
    
    for source, count in sources.items():
        filename = source.split('\\')[-1] if '\\' in source else source
        print(f"  {filename}: {count} chunks")
    
    # Validation
    print("\n✅ VALIDATION:")
    checks = {
        "Min size >= 50": min(chunk_lengths) >= 50,
        "Max size <= 2000": max(chunk_lengths) <= 2000,
        "No empty chunks": all(len(c.page_content) > 0 for c in chunks),
        "All have metadata": all(c.metadata for c in chunks),
    }
    
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"  {status} {check}")
    
    return all(checks.values())


if __name__ == "__main__":
    from loader import load_pdfs
    
    pdf_paths = [
        "data/raw/Machine_Learning.pdf",
        "data/raw/Deep_Learning.pdf",
    ]
    
    try:
        # Load documents
        documents = load_pdfs(pdf_paths)
        
        # Split documents
        chunks = split_documents(documents, min_chunk_size=100)
        
        # Analyze chunks
        is_valid = analyze_chunks(chunks)
        
        # Show first chunk
        print("\n" + "="*50)
        print("FIRST CHUNK SAMPLE")
        print("="*50)
        print(chunks[0].page_content[:500])
        print("\n📄 Metadata:", chunks[0].metadata.get('source', 'unknown'))
        
        if is_valid:
            print("\n✅ Ready for embeddings and vector storage!")
        else:
            print("\n⚠️  Review chunk quality before proceeding")
            
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        exit(1)