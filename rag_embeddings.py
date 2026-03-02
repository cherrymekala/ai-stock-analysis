import os
import json
from typing import List, Dict, Optional
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  Optional embedding libraries not installed. Install with: pip install sentence-transformers faiss-cpu")

class FinancialDocumentRAG:
    """RAG system for financial documents"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize RAG with embedding model"""
        if not EMBEDDINGS_AVAILABLE:
            raise RuntimeError("sentence-transformers and faiss-cpu required. Install: pip install sentence-transformers faiss-cpu")
        
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None
        self.faiss_index = None
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def add_documents(self, documents: List[Dict[str, str]]):
        """
        Add documents to the RAG system
        Each document should have 'content' and optionally 'metadata'
        """
        self.documents = documents

        contents = [doc.get("content", "") for doc in documents]
        embeddings = self.model.encode(contents, convert_to_numpy=True).astype(np.float32)

        faiss.normalize_L2(embeddings)

        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(embeddings)
        self.embeddings = embeddings
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve most relevant documents for a query
        Returns top_k most similar documents
        """
        if not self.faiss_index:
            return []

        query_embedding = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_embedding)

        similarities, indices = self.faiss_index.search(query_embedding, min(top_k, len(self.documents)))

        results = []
        for idx, similarity in zip(indices[0], similarities[0]):
            doc = self.documents[int(idx)]

            results.append({
                **doc,
                "similarity_score": float(similarity),
            })
        
        return results
    
    def generate_rag_context(self, query: str, top_k: int = 3) -> str:
        """Generate context string from retrieved documents for LLM"""
        relevant_docs = self.retrieve_relevant_docs(query, top_k)
        
        if not relevant_docs:
            return ""
        
        context = "**Relevant Financial Documents:**\n\n"
        for i, doc in enumerate(relevant_docs, 1):
            context += f"[Document {i}] (Relevance: {doc.get('similarity_score', 0):.1%})\n"
            context += f"Source: {doc.get('metadata', {}).get('source', 'Unknown')}\n"
            context += f"Content: {doc.get('content', '')[:500]}...\n\n"
        
        return context

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks for embedding"""
    chunks = []
    words = text.split()
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def prepare_filing_documents(filing_data: Dict) -> List[Dict]:
    """
    Convert SEC filing data into document chunks for RAG
    Prioritizes extracted sections over raw text to avoid embedding junk content
    """
    if not filing_data:
        return []
    
    documents = []
    ticker = filing_data.get("ticker", "")
    filing_type = filing_data.get("filing_type", "")
    date = filing_data.get("date", "")
    url = filing_data.get("url", "")

    sections = filing_data.get("sections", {})
    
    if sections:

        section_names = {
            "business": "Business Overview",
            "risks": "Risk Factors",
            "md_and_a": "Management, Discussion & Analysis"
        }
        
        chunk_index = 0
        for section_key, section_name in section_names.items():
            if section_key in sections:
                section_text = sections[section_key]

                chunks = chunk_text(section_text, chunk_size=500, overlap=50)
                
                for chunk in chunks:
                    documents.append({
                        "content": chunk,
                        "metadata": {
                            "source": f"{ticker} {filing_type} - {section_name} ({date})",
                            "section": section_name,
                            "filing_type": filing_type,
                            "date": date,
                            "ticker": ticker,
                            "chunk_index": chunk_index,
                            "url": url,
                        }
                    })
                    chunk_index += 1
    else:

        text = filing_data.get("text", "")
        if text:
            chunks = chunk_text(text, chunk_size=500, overlap=50)
            for i, chunk in enumerate(chunks):
                documents.append({
                    "content": chunk,
                    "metadata": {
                        "source": f"{ticker} {filing_type} ({date})",
                        "filing_type": filing_type,
                        "date": date,
                        "ticker": ticker,
                        "chunk_index": i,
                        "url": url,
                    }
                })
    
    return documents

def create_filing_rag(ticker: str, filings: List[Dict]) -> Optional['FinancialDocumentRAG']:
    """
    Create a RAG system from SEC filings
    """
    if not EMBEDDINGS_AVAILABLE:
        return None
    
    try:
        rag = FinancialDocumentRAG()

        all_documents = []
        for filing in filings:
            docs = prepare_filing_documents(filing)
            all_documents.extend(docs)
        
        if all_documents:
            rag.add_documents(all_documents)
            print(f"✅ Created RAG system with {len(all_documents)} document chunks")
            return rag
        else:
            print("⚠️  No documents to add to RAG")
            return None
    
    except Exception as e:
        print(f"Error creating RAG system: {e}")
        return None
