from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader

def load_pdfs(pdf_paths: list[str]):
    """
    Load multiple PDF files and return a list of LangChain Documents.
    """

    all_documents = []

    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            print(f"❌ File not found: {pdf_path}")
            continue

        print(f"📄 Loading: {pdf_path.name}")

        loader = PyMuPDFLoader(str(pdf_path))
        documents = loader.load()

        print(f"✅ Pages loaded: {len(documents)}")

        all_documents.extend(documents)

    return all_documents


if __name__ == "__main__":

    pdf_paths = [
        "data/raw/Machine_Learning.pdf",
        "data/raw/Deep_Learning.pdf",
    ]

    documents = load_pdfs(pdf_paths)

    print("\n==============================")
    print("TOTAL DOCUMENTS/PAGES LOADED")
    print("==============================")
    print(f"Total pages: {len(documents)}")

    print("\n==============================")
    print("FIRST DOCUMENT")
    print("==============================")
    print(documents[0].page_content[:1000])

    print("\nMetadata:")
    print(documents[0].metadata)