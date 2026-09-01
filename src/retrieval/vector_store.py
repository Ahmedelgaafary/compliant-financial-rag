"""
Persistent vector store with FAISS support.
Run: python -c "from src.retrieval.vector_store import VectorStore"
"""

import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.ingestion.chunker import DocumentChunk
from src.retrieval.models import RetrievalResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Semantic vector retrieval using sentence-transformer embeddings.
    Supports persistence with FAISS indexing.
    """

    def __init__(
        self,
        chunks: Optional[Tuple[DocumentChunk, ...]] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_dir: Optional[str] = None,
        force_rebuild: bool = False,
    ):
        """
        Initialize the vector store.
        
        Args:
            chunks: Document chunks to index (optional if loading existing)
            model_name: Sentence transformer model name
            index_dir: Directory to store/load index (default: data/indexes)
            force_rebuild: Force rebuild even if index exists
        """
        self.chunks = chunks
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        
        # Set index directory
        if index_dir is None:
            index_dir = "data/indexes"
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Paths for persistence
        self.index_path = self.index_dir / "vector_index.faiss"
        self.chunks_path = self.index_dir / "vector_chunks.pkl"
        self.metadata_path = self.index_dir / "vector_metadata.json"
        
        # Load or build index
        if chunks:
            # Build new index from chunks
            self._build_index(force_rebuild=force_rebuild)
        else:
            # Try to load existing index
            self._load_index()
    
    def _build_index(self, force_rebuild: bool = False):
        """Build and save FAISS index from chunks."""
        if not self.chunks:
            raise ValueError("Cannot build index: no chunks provided")
        
        # Check if existing index can be loaded
        if not force_rebuild and self._load_existing_index():
            logger.info("✅ Loaded existing index, skipping rebuild")
            return
        
        logger.info(f"🔄 Building vector index for {len(self.chunks)} chunks...")
        
        # Encode all chunks
        texts = [chunk.text for chunk in self.chunks]
        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        
        # Try to import faiss
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not installed. Using in-memory only.")
            logger.info("To install FAISS: pip install faiss-cpu")
            return
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
        self.index.add(self.embeddings.astype(np.float32))
        
        # Save to disk
        self._save_index()
        logger.info(f"✅ Index built and saved to {self.index_dir}")
    
    def _save_index(self):
        """Save index and chunks to disk."""
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not installed. Cannot save index.")
            return
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path))
        
        # Save chunks
        with open(self.chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "num_chunks": len(self.chunks),
            "dimension": self.index.d if hasattr(self, 'index') else None,
            "chunks_path": str(self.chunks_path),
        }
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Saved index: {self.index_path}")
    
    def _load_existing_index(self) -> bool:
        """Load existing index from disk."""
        try:
            import faiss
        except ImportError:
            return False
        
        if not self.index_path.exists() or not self.chunks_path.exists():
            return False
        
        try:
            logger.info("📂 Loading existing vector index...")
            self.index = faiss.read_index(str(self.index_path))
            with open(self.chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"   Model: {metadata.get('model_name', 'unknown')}")
                logger.info(f"   Chunks: {metadata.get('num_chunks', 0)}")
                logger.info(f"   Dimension: {metadata.get('dimension', 0)}")
            
            logger.info(f"✅ Index loaded: {len(self.chunks)} chunks")
            return True
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            return False
    
    def _load_index(self):
        """Load existing index or warn if not found."""
        if not self._load_existing_index():
            logger.warning("No existing index found. Initialize with chunks or run reindex.py")
            self.index = None
            self.chunks = ()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> Tuple[RetrievalResult, ...]:
        """Retrieve semantically relevant chunks."""
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        
        # Check if index exists
        if not hasattr(self, 'index') or self.index is None:
            logger.warning("No index loaded. Falling back to in-memory encoding.")
            return self._retrieve_in_memory(query, top_k)
        
        # Encode query
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        
        # Search FAISS index
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            min(top_k, len(self.chunks)),
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append(
                    RetrievalResult(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        text=chunk.text,
                        score=float(score),
                        page_number=chunk.page_number,
                        section=chunk.section,
                        document_sha256=chunk.document_sha256,
                        retrieval_method="vector",
                    )
                )
        
        logger.info(f"Vector retrieval: query='{query[:50]}...', results={len(results)}")
        return tuple(results)
    
    def _retrieve_in_memory(
        self,
        query: str,
        top_k: int = 5,
    ) -> Tuple[RetrievalResult, ...]:
        """Fallback: in-memory retrieval without FAISS."""
        if not self.chunks:
            return ()
        
        # Encode query and all chunks
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        
        # Compute similarities (in-memory)
        import numpy as np
        chunk_embeddings = self.model.encode(
            [chunk.text for chunk in self.chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        
        scores = np.dot(chunk_embeddings, query_embedding)
        indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in indices:
            chunk = self.chunks[idx]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=float(scores[idx]),
                    page_number=chunk.page_number,
                    section=chunk.section,
                    document_sha256=chunk.document_sha256,
                    retrieval_method="vector",
                )
            )
        
        return tuple(results)
    
    def get_stats(self) -> dict:
        """Get index statistics."""
        stats = {
            "index_dir": str(self.index_dir),
            "model_name": self.model_name,
            "num_chunks": len(self.chunks) if self.chunks else 0,
            "has_index": hasattr(self, 'index') and self.index is not None,
        }
        
        if hasattr(self, 'index') and self.index is not None:
            stats.update({
                "dimension": self.index.d,
                "index_type": type(self.index).__name__,
                "index_path": str(self.index_path),
                "chunks_path": str(self.chunks_path),
            })
        
        return stats


def rebuild_vector_index():
    """Utility function to rebuild the vector index."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.ingestion.pipeline import FinancialDocumentPipeline
    
    # Find all PDFs
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return
    
    pdfs = list(data_dir.glob("*.pdf"))
    if not pdfs:
        print(f"❌ No PDFs found in {data_dir}")
        return
    
    print(f"📄 Found {len(pdfs)} PDFs")
    
    # Process all documents
    pipeline = FinancialDocumentPipeline()
    all_chunks = []
    
    for pdf in pdfs:
        print(f"📖 Processing: {pdf.name}")
        result = pipeline.process(pdf)
        all_chunks.extend(result.chunks)
        print(f"   ✅ {len(result.chunks)} chunks")
    
    print(f"📊 Total chunks: {len(all_chunks)}")
    
    # Build and save index
    vector_store = VectorStore(
        chunks=tuple(all_chunks),
        force_rebuild=True,
    )
    print("✅ Index built and saved!")
    print(vector_store.get_stats())


if __name__ == "__main__":
    rebuild_vector_index()