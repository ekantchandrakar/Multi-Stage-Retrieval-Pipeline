"""
Data Processor for loading and preprocessing the search corpus
"""
import pandas as pd
from typing import List, Dict, Any, Tuple
from pathlib import Path

from .utils import clean_text, setup_logger

logger = setup_logger(__name__)


class DataProcessor:
    """
    Handles data loading, cleaning, and preprocessing for the search system.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize DataProcessor.
        
        Args:
            data_path: Path to the CSV dataset
        """
        self.data_path = Path(data_path)
        self.df = None
        self.corpus = []
        self.metadata = []
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            Loaded DataFrame
        """
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.df)} documents")
            logger.info(f"Columns: {list(self.df.columns)}")
            
            return self.df
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, 
                       combine_fields: List[str] = None,
                       clean: bool = True) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Preprocess data and create searchable corpus.
        
        Args:
            combine_fields: Fields to combine for searchable text (default: ['title', 'body'])
            clean: Whether to clean text
            
        Returns:
            Tuple of (corpus, metadata)
        """
        if self.df is None:
            self.load_data()
        
        if combine_fields is None:
            combine_fields = ['title', 'body']
        
        logger.info(f"Preprocessing data with fields: {combine_fields}")
        
        # Combine text fields
        corpus = []
        metadata = []
        
        for idx, row in self.df.iterrows():
            # Combine specified fields
            text_parts = []
            for field in combine_fields:
                if field in row and pd.notna(row[field]):
                    text_parts.append(str(row[field]))
            
            combined_text = " ".join(text_parts)
            
            # Clean text if requested
            if clean:
                combined_text = clean_text(combined_text)
            
            corpus.append(combined_text)
            
            # Store metadata
            meta = {
                'id': row.get('id', idx),
                'category': row.get('category', ''),
                'difficulty': row.get('difficulty', ''),
                'title': row.get('title', ''),
                'body': row.get('body', ''),
                'tags': row.get('tags', ''),
                'index': idx
            }
            metadata.append(meta)
        
        self.corpus = corpus
        self.metadata = metadata
        
        logger.info(f"Preprocessed {len(corpus)} documents")
        
        return corpus, metadata
    
    def get_corpus(self) -> List[str]:
        """
        Get the preprocessed corpus.
        
        Returns:
            List of document texts
        """
        if not self.corpus:
            self.preprocess_data()
        return self.corpus
    
    def get_metadata(self) -> List[Dict[str, Any]]:
        """
        Get document metadata.
        
        Returns:
            List of metadata dictionaries
        """
        if not self.metadata:
            self.preprocess_data()
        return self.metadata
    
    def get_document_by_index(self, index: int) -> Dict[str, Any]:
        """
        Get document by index.
        
        Args:
            index: Document index
            
        Returns:
            Document metadata
        """
        if not self.metadata:
            self.preprocess_data()
        
        if 0 <= index < len(self.metadata):
            return self.metadata[index]
        else:
            raise IndexError(f"Document index {index} out of range")
    
    def get_document_by_id(self, doc_id: str) -> Dict[str, Any]:
        """
        Get document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document metadata
        """
        if not self.metadata:
            self.preprocess_data()
        
        for meta in self.metadata:
            if meta['id'] == doc_id:
                return meta
        
        raise ValueError(f"Document with ID {doc_id} not found")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary of statistics
        """
        if self.df is None:
            self.load_data()
        
        stats = {
            'total_documents': len(self.df),
            'categories': {},
            'difficulty_levels': {},
            'avg_text_length': 0,
        }
        
        # Category distribution
        if 'category' in self.df.columns:
            stats['categories'] = self.df['category'].value_counts().to_dict()
        
        # Difficulty distribution
        if 'difficulty' in self.df.columns:
            stats['difficulty_levels'] = self.df['difficulty'].value_counts().to_dict()
        
        # Average text length
        if self.corpus:
            stats['avg_text_length'] = sum(len(text) for text in self.corpus) / len(self.corpus)
        
        return stats
    
    def filter_by_category(self, category: str) -> List[int]:
        """
        Get document indices by category.
        
        Args:
            category: Category name
            
        Returns:
            List of document indices
        """
        if not self.metadata:
            self.preprocess_data()
        
        return [i for i, meta in enumerate(self.metadata) 
                if meta['category'].lower() == category.lower()]
    
    def filter_by_difficulty(self, difficulty: str) -> List[int]:
        """
        Get document indices by difficulty.
        
        Args:
            difficulty: Difficulty level
            
        Returns:
            List of document indices
        """
        if not self.metadata:
            self.preprocess_data()
        
        return [i for i, meta in enumerate(self.metadata) 
                if meta['difficulty'].lower() == difficulty.lower()]
    
    def search_by_tags(self, tag: str) -> List[int]:
        """
        Get document indices by tag.
        
        Args:
            tag: Tag to search for
            
        Returns:
            List of document indices
        """
        if not self.metadata:
            self.preprocess_data()
        
        tag_lower = tag.lower()
        return [i for i, meta in enumerate(self.metadata) 
                if tag_lower in meta['tags'].lower()]
    
    def __len__(self) -> int:
        """Get number of documents."""
        if self.df is None:
            self.load_data()
        return len(self.df)
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get document by index."""
        return self.get_document_by_index(index)
