#!/usr/bin/env python
"""
Re-index all financial documents including newly added PDFs.
Run: python reindex.py
"""

import pickle
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.ingestion.pipeline import FinancialDocumentPipeline
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.vector_store import VectorStore


def main():
    print("=" * 70)
    print("📚 RE-INDEXING ALL FINANCIAL DOCUMENTS")
    print("=" * 70)

    # 1. Find all PDFs
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        print("   Create: mkdir data/raw")
        return

    pdfs = list(data_dir.glob("*.pdf"))
    if not pdfs:
        print(f"❌ No PDFs found in {data_dir}")
        print("   Add PDF files to data/raw/")
        return

    print(f"\n📄 Found {len(pdfs)} PDFs:")
    for pdf in pdfs:
        size = pdf.stat().st_size / 1024 / 1024
        print(f"   - {pdf.name} ({size:.1f} MB)")

    # 2. Process all documents
    print("\n🔄 Processing documents...")
    pipeline = FinancialDocumentPipeline()
    all_chunks = []

    for pdf in pdfs:
        print(f"\n📖 Processing: {pdf.name}")
        try:
            result = pipeline.process(pdf)
            chunks = list(result.chunks)
            all_chunks.extend(chunks)
            print(f"   ✅ Created {len(chunks)} chunks")
            print(f"   📄 Document ID: {result.document.document_id}")
            print(f"   🔑 SHA256: {result.document.sha256[:16]}...")
        except Exception as e:
            print(f"   ❌ Error: {e}")

    if not all_chunks:
        print("\n❌ No chunks created!")
        return

    print(f"\n📊 Total chunks: {len(all_chunks)}")

    # 3. Clear old indexes (optional)
    index_dir = Path("data/indexes")
    if index_dir.exists():
        shutil.rmtree(index_dir)
        print("🗑️  Removed old indexes")
    
    index_dir.mkdir(parents=True, exist_ok=True)

    # 4. Build BM25 index
    print("\n🔍 Building BM25 index...")
    bm25_retriever = BM25Retriever(tuple(all_chunks))
    print("   ✅ BM25 index built")

    # 5. Build Vector index (with persistence)
    print("\n🧠 Building Vector index...")
    vector_store = VectorStore(
        chunks=tuple(all_chunks),
        index_dir=str(index_dir),
        force_rebuild=True,
    )
    print("   ✅ Vector index built")
    print(f"   Stats: {vector_store.get_stats()}")

    # 6. Build Hybrid retriever
    print("\n🔀 Building Hybrid retriever...")
    hybrid_retriever = HybridRetriever(
        bm25_retriever=bm25_retriever,
        vector_store=vector_store,
    )
    print("   ✅ Hybrid retriever built")

    # 7. Save chunk metadata
    chunks_path = index_dir / "chunks_metadata.pkl"
    with open(chunks_path, 'wb') as f:
        pickle.dump({
            "total_chunks": len(all_chunks),
            "documents": list(set(c.document_id for c in all_chunks)),
        }, f)
    print(f"   💾 Metadata saved to {chunks_path}")

    # 8. Test retrieval for each company
    print("\n🔍 TESTING RETRIEVAL:")
    test_queries = [
        ("Apple", "Apple revenue 2025"),
        ("Microsoft", "Microsoft revenue 2025"),
        ("Real Brokerage", "Real Brokerage revenue 2025"),
    ]
    
    for company, query in test_queries:
        results = hybrid_retriever.retrieve(query, top_k=3)
        if results:
            print(f"   ✅ {company}: Found {len(results)} results")
            for i, r in enumerate(results[:2], 1):
                doc = r.document_id
                text_preview = r.text[:80].replace('\n', ' ')
                print(f"      {i}. {doc}: {text_preview}...")
        else:
            print(f"   ⚠️  {company}: No results")

    print("\n" + "=" * 70)
    print("✅ RE-INDEXING COMPLETE!")
    print("=" * 70)
    print(f"   Documents: {len(pdfs)}")
    print(f"   Total chunks: {len(all_chunks)}")
    print("   BM25: Ready")
    print("   Vector: Ready")
    print("   Hybrid: Ready")
    print("\n🚀 System is ready for queries!")


if __name__ == "__main__":
    main()