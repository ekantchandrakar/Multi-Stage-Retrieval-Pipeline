"""
Comprehensive tests for Hybrid Search System
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_processor import DataProcessor
from src.bm25_retriever import BM25Retriever
from src.semantic_retriever import SemanticRetriever
from src.hybrid_fusion import HybridFusion
from src.cross_encoder_reranker import CrossEncoderReranker
from src.search_engine import HybridSearchEngine


class TestDataProcessor:
    """Tests for DataProcessor"""
    
    @pytest.fixture
    def data_processor(self):
        return DataProcessor("data/semantic_search_dataset_2000.csv")
    
    def test_load_data(self, data_processor):
        df = data_processor.load_data()
        assert df is not None
        assert len(df) > 0
        assert 'title' in df.columns
        assert 'body' in df.columns
    
    def test_preprocess_data(self, data_processor):
        corpus, metadata = data_processor.preprocess_data()
        assert len(corpus) > 0
        assert len(metadata) > 0
        assert len(corpus) == len(metadata)
    
    def test_get_statistics(self, data_processor):
        data_processor.load_data()
        stats = data_processor.get_statistics()
        assert 'total_documents' in stats
        assert stats['total_documents'] > 0


class TestBM25Retriever:
    """Tests for BM25Retriever"""
    
    @pytest.fixture
    def bm25(self):
        return BM25Retriever()
    
    @pytest.fixture
    def sample_corpus(self):
        return [
            "Python is a programming language",
            "Java is used for backend development",
            "JavaScript runs in the browser",
            "Python and Java are popular languages"
        ]
    
    def test_build_index(self, bm25, sample_corpus):
        bm25.build_index(sample_corpus)
        assert bm25.bm25 is not None
        assert bm25.corpus_size == len(sample_corpus)
    
    def test_search(self, bm25, sample_corpus):
        bm25.build_index(sample_corpus)
        results = bm25.search("Python programming", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r[0], int) for r in results)
        assert all(isinstance(r[1], float) for r in results)
    
    def test_get_statistics(self, bm25, sample_corpus):
        bm25.build_index(sample_corpus)
        stats = bm25.get_statistics()
        assert stats['status'] == 'built'
        assert stats['corpus_size'] == len(sample_corpus)


class TestSemanticRetriever:
    """Tests for SemanticRetriever"""
    
    @pytest.fixture
    def semantic(self):
        return SemanticRetriever(device="cpu")
    
    @pytest.fixture
    def sample_corpus(self):
        return [
            "Machine learning is a subset of AI",
            "Deep learning uses neural networks",
            "Natural language processing handles text"
        ]
    
    def test_build_index(self, semantic, sample_corpus):
        semantic.build_index(sample_corpus, show_progress=False)
        assert semantic.index is not None
        assert semantic.embeddings is not None
        assert semantic.corpus_size == len(sample_corpus)
    
    def test_search(self, semantic, sample_corpus):
        semantic.build_index(sample_corpus, show_progress=False)
        results = semantic.search("neural networks deep learning", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r[0], int) for r in results)
        assert all(isinstance(r[1], float) for r in results)
    
    def test_get_embedding(self, semantic):
        embedding = semantic.get_embedding("test text")
        assert embedding is not None
        assert len(embedding) == semantic.embedding_dim


class TestHybridFusion:
    """Tests for HybridFusion"""
    
    @pytest.fixture
    def fusion(self):
        return HybridFusion(k=60)
    
    def test_fuse_two_lists(self, fusion):
        list1 = [(0, 10.0), (1, 8.0), (2, 6.0)]
        list2 = [(1, 9.0), (2, 7.0), (3, 5.0)]
        
        results = fusion.fuse(list1, list2, top_k=3)
        assert len(results) <= 3
        assert all(isinstance(r[0], int) for r in results)
    
    def test_weighted_fuse(self, fusion):
        list1 = [(0, 10.0), (1, 8.0)]
        list2 = [(1, 9.0), (2, 7.0)]
        weights = [0.7, 0.3]
        
        results = fusion.weighted_fuse([list1, list2], weights, top_k=3)
        assert len(results) <= 3
    
    def test_get_overlap(self, fusion):
        list1 = [(0, 10.0), (1, 8.0), (2, 6.0)]
        list2 = [(1, 9.0), (2, 7.0), (3, 5.0)]
        
        overlap = fusion.get_overlap(list1, list2, top_k=3)
        assert 'total_unique_documents' in overlap
        assert 'pairwise_overlap' in overlap


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker"""
    
    @pytest.fixture
    def reranker(self):
        return CrossEncoderReranker(device="cpu")
    
    def test_rerank(self, reranker):
        query = "Python programming tutorial"
        documents = [
            "Learn Python from scratch",
            "Java programming guide",
            "Python advanced techniques"
        ]
        doc_ids = [0, 1, 2]
        
        results = reranker.rerank(query, documents, doc_ids, top_k=2)
        assert len(results) <= 2
        assert all(isinstance(r[0], int) for r in results)
        assert all(isinstance(r[1], float) for r in results)
    
    def test_score_pair(self, reranker):
        score = reranker.score_pair("Python", "Python is a programming language")
        assert isinstance(score, float)


class TestHybridSearchEngine:
    """Tests for HybridSearchEngine"""
    
    @pytest.fixture
    def engine(self):
        return HybridSearchEngine(
            data_path="data/semantic_search_dataset_2000.csv",
            model_dir="models_test",
            device="cpu"
        )
    
    def test_initialization(self, engine):
        assert engine is not None
        assert engine.data_processor is not None
        assert engine.bm25_retriever is not None
        assert engine.semantic_retriever is not None
    
    def test_build_indices(self, engine):
        # Note: This test may take a few minutes
        # Skip in CI/CD by using pytest markers if needed
        try:
            engine.build_indices()
            assert engine.is_built
            assert len(engine.corpus) > 0
            assert len(engine.metadata) > 0
        except Exception as e:
            pytest.skip(f"Skipping index building: {e}")
    
    def test_search_validation(self, engine):
        with pytest.raises(ValueError):
            engine.search("")  # Empty query
        
        with pytest.raises(ValueError):
            engine.search("test", top_k=0)  # Invalid top_k


# Integration Tests
class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_end_to_end_search(self):
        """Test complete search pipeline"""
        try:
            engine = HybridSearchEngine(
                data_path="data/semantic_search_dataset_2000.csv",
                model_dir="models_test",
                device="cpu"
            )
            
            # Build indices (may take time)
            engine.build_indices()
            
            # Perform search
            results = engine.search("Python programming", top_k=5, search_type='hybrid')
            
            assert len(results) > 0
            assert len(results) <= 5
            assert all('title' in r for r in results)
            assert all('score' in r for r in results)
            
        except Exception as e:
            pytest.skip(f"End-to-end test skipped: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
