"""
Vector Store Utilities for DOSO AI

This module provides a FAISS-based vector store implementation for
storing and retrieving embeddings. It includes utilities for creating
embeddings using OpenAI's embeddings API and searching for similar vectors.
"""

import os
import json
import pickle
import logging
import tempfile
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import openai
from openai_agents import Agent, RunConfig, run_agent, function_tool
from ..config import settings
from ..monitoring.tracing import trace_method, trace_async_method

logger = logging.getLogger(__name__)

# Default embedding dimensions for OpenAI models
EMBEDDING_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}

class VectorStore:
    """
    FAISS-based vector store implementation
    
    This class provides a vector store implementation using FAISS for
    storing and retrieving embeddings. It handles persistence to disk
    and supports multiple indexes for different types of data.
    """
    
    def __init__(
        self, 
        base_path: str = "doso-ai/data/vector_store",
        embedding_model: str = "text-embedding-3-large",
        create_if_missing: bool = True
    ):
        """
        Initialize the vector store
        
        Args:
            base_path: Base path for storing vector indexes
            embedding_model: OpenAI embedding model to use
            create_if_missing: Whether to create the store if it doesn't exist
        """
        self.base_path = base_path
        self.embedding_model = embedding_model
        self.embedding_dim = EMBEDDING_DIMENSIONS.get(embedding_model, 1536)
        
        # Ensure base directory exists
        if create_if_missing:
            os.makedirs(base_path, exist_ok=True)
            
        # Initialize FAISS indexes
        self.indexes: Dict[str, faiss.Index] = {}
        self.metadata: Dict[str, List[Dict[str, Any]]] = {}
        
        # Load existing indexes if available
        self._load_indexes()
    
    def _get_index_path(self, index_name: str) -> str:
        """Get path for a FAISS index file"""
        return os.path.join(self.base_path, f"{index_name}.faiss")
        
    def _get_metadata_path(self, index_name: str) -> str:
        """Get path for a metadata file"""
        return os.path.join(self.base_path, f"{index_name}.metadata.json")
    
    @trace_method("vector_store.create_embedding")
    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding using OpenAI's embeddings API
        
        Args:
            text: Text to create an embedding for
            
        Returns:
            List of floats representing the embedding
        """
        response = openai.embeddings.create(
            model=self.embedding_model,
            input=text,
            encoding_format="float"
        )
        
        embedding = response.data[0].embedding
        return embedding
    
    @trace_method("vector_store.batch_create_embeddings")
    def batch_create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for multiple texts
        
        Args:
            texts: List of texts to create embeddings for
            
        Returns:
            List of embeddings for each text
        """
        if not texts:
            return []
            
        response = openai.embeddings.create(
            model=self.embedding_model,
            input=texts,
            encoding_format="float"
        )
        
        embeddings = [data.embedding for data in response.data]
        return embeddings
    
    @trace_method("vector_store._load_indexes")
    def _load_indexes(self) -> None:
        """Load existing FAISS indexes from disk"""
        for filename in os.listdir(self.base_path):
            if filename.endswith(".faiss"):
                index_name = filename.replace(".faiss", "")
                self.load_index(index_name)
    
    @trace_method("vector_store.load_index")
    def load_index(self, index_name: str) -> bool:
        """
        Load a specific index from disk
        
        Args:
            index_name: Name of the index to load
            
        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = self._get_index_path(index_name)
        metadata_path = self._get_metadata_path(index_name)
        
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.warning(f"Index {index_name} not found on disk")
            return False
            
        try:
            # Load the FAISS index
            self.indexes[index_name] = faiss.read_index(index_path)
            
            # Load the metadata
            with open(metadata_path, "r") as f:
                self.metadata[index_name] = json.load(f)
                
            logger.info(f"Loaded index {index_name} with {self.indexes[index_name].ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index {index_name}: {str(e)}")
            return False
    
    @trace_method("vector_store.create_index")
    def create_index(self, index_name: str) -> None:
        """
        Create a new FAISS index
        
        Args:
            index_name: Name of the index to create
        """
        # Create a new FAISS index
        index = faiss.IndexFlatL2(self.embedding_dim)
        self.indexes[index_name] = index
        self.metadata[index_name] = []
        
        # Save the empty index
        self.save_index(index_name)
        
        logger.info(f"Created new index {index_name}")
    
    @trace_method("vector_store.save_index")
    def save_index(self, index_name: str) -> bool:
        """
        Save an index to disk
        
        Args:
            index_name: Name of the index to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        if index_name not in self.indexes:
            logger.warning(f"Index {index_name} not found in memory")
            return False
            
        index_path = self._get_index_path(index_name)
        metadata_path = self._get_metadata_path(index_name)
        
        try:
            # Save the FAISS index
            faiss.write_index(self.indexes[index_name], index_path)
            
            # Save the metadata
            with open(metadata_path, "w") as f:
                json.dump(self.metadata[index_name], f, indent=2, default=str)
                
            logger.info(f"Saved index {index_name} with {self.indexes[index_name].ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error saving index {index_name}: {str(e)}")
            return False
    
    @trace_method("vector_store.add_text")
    def add_text(
        self, 
        index_name: str, 
        text: str,
        metadata: Dict[str, Any],
        create_if_missing: bool = True
    ) -> int:
        """
        Add text to an index
        
        Args:
            index_name: Name of the index to add to
            text: Text to add
            metadata: Metadata to associate with the text
            create_if_missing: Whether to create the index if it doesn't exist
            
        Returns:
            ID of the added vector
        """
        # Create index if needed
        if index_name not in self.indexes:
            if create_if_missing:
                self.create_index(index_name)
            else:
                raise ValueError(f"Index {index_name} not found")
        
        # Create embedding
        embedding = self.create_embedding(text)
        vector = np.array([embedding], dtype=np.float32)
        
        # Add to index
        vector_id = self.indexes[index_name].ntotal
        self.indexes[index_name].add(vector)
        
        # Add metadata
        metadata_entry = {"id": vector_id, "text": text, **metadata}
        self.metadata[index_name].append(metadata_entry)
        
        # Save changes
        self.save_index(index_name)
        
        return vector_id
    
    @trace_method("vector_store.batch_add_texts")
    def batch_add_texts(
        self,
        index_name: str,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        create_if_missing: bool = True
    ) -> List[int]:
        """
        Add multiple texts to an index
        
        Args:
            index_name: Name of the index to add to
            texts: List of texts to add
            metadatas: List of metadata for each text
            create_if_missing: Whether to create the index if it doesn't exist
            
        Returns:
            List of IDs for the added vectors
        """
        if len(texts) != len(metadatas):
            raise ValueError("Length of texts and metadatas must match")
            
        if not texts:
            return []
            
        # Create index if needed
        if index_name not in self.indexes:
            if create_if_missing:
                self.create_index(index_name)
            else:
                raise ValueError(f"Index {index_name} not found")
        
        # Create embeddings
        embeddings = self.batch_create_embeddings(texts)
        vectors = np.array(embeddings, dtype=np.float32)
        
        # Add to index
        start_id = self.indexes[index_name].ntotal
        self.indexes[index_name].add(vectors)
        
        # Add metadata
        vector_ids = list(range(start_id, start_id + len(texts)))
        for i, (text, metadata, vector_id) in enumerate(zip(texts, metadatas, vector_ids)):
            metadata_entry = {"id": vector_id, "text": text, **metadata}
            self.metadata[index_name].append(metadata_entry)
        
        # Save changes
        self.save_index(index_name)
        
        return vector_ids
    
    @trace_method("vector_store.search")
    def search(
        self,
        index_name: str,
        query_text: str,
        top_k: int = 5,
        include_metadata: bool = True,
        filter_fn: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in an index
        
        Args:
            index_name: Name of the index to search
            query_text: Text to search for
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results
            filter_fn: Optional function to filter results based on metadata
            
        Returns:
            List of search results, each with id, score, text, and metadata
        """
        if index_name not in self.indexes:
            logger.warning(f"Index {index_name} not found")
            return []
            
        # Create embedding for query
        query_embedding = self.create_embedding(query_text)
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Search the index
        scores, vector_ids = self.indexes[index_name].search(query_vector, top_k)
        
        # Prepare results
        results = []
        for i, (score, vector_id) in enumerate(zip(scores[0], vector_ids[0])):
            # Skip invalid IDs (happens if there are fewer results than top_k)
            if vector_id == -1:
                continue
                
            # Get metadata
            metadata_entry = self.metadata[index_name][vector_id]
            
            # Apply filter if provided
            if filter_fn and not filter_fn(metadata_entry):
                continue
                
            # Add result
            result = {
                "id": int(vector_id),
                "score": float(score),
                "text": metadata_entry["text"]
            }
            
            # Include metadata if requested
            if include_metadata:
                result["metadata"] = {k: v for k, v in metadata_entry.items() if k != "text"}
                
            results.append(result)
            
        return results
    
    @trace_method("vector_store.search_by_embedding")
    def search_by_embedding(
        self,
        index_name: str,
        embedding: List[float],
        top_k: int = 5,
        include_metadata: bool = True,
        filter_fn: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors using a pre-computed embedding
        
        Args:
            index_name: Name of the index to search
            embedding: Pre-computed embedding to search for
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results
            filter_fn: Optional function to filter results based on metadata
            
        Returns:
            List of search results, each with id, score, text, and metadata
        """
        if index_name not in self.indexes:
            logger.warning(f"Index {index_name} not found")
            return []
            
        # Convert embedding to numpy array
        query_vector = np.array([embedding], dtype=np.float32)
        
        # Search the index
        scores, vector_ids = self.indexes[index_name].search(query_vector, top_k)
        
        # Prepare results (similar to search method)
        results = []
        for i, (score, vector_id) in enumerate(zip(scores[0], vector_ids[0])):
            if vector_id == -1:
                continue
                
            metadata_entry = self.metadata[index_name][vector_id]
            
            if filter_fn and not filter_fn(metadata_entry):
                continue
                
            result = {
                "id": int(vector_id),
                "score": float(score),
                "text": metadata_entry["text"]
            }
            
            if include_metadata:
                result["metadata"] = {k: v for k, v in metadata_entry.items() if k != "text"}
                
            results.append(result)
            
        return results
    
    @trace_method("vector_store.delete_index")
    def delete_index(self, index_name: str) -> bool:
        """
        Delete an index from disk and memory
        
        Args:
            index_name: Name of the index to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        index_path = self._get_index_path(index_name)
        metadata_path = self._get_metadata_path(index_name)
        
        try:
            # Remove from memory
            if index_name in self.indexes:
                del self.indexes[index_name]
                
            if index_name in self.metadata:
                del self.metadata[index_name]
                
            # Remove from disk
            if os.path.exists(index_path):
                os.remove(index_path)
                
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                
            logger.info(f"Deleted index {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting index {index_name}: {str(e)}")
            return False
    
    @trace_method("vector_store.get_metadata")
    def get_metadata(self, index_name: str, vector_id: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific vector
        
        Args:
            index_name: Name of the index
            vector_id: ID of the vector
            
        Returns:
            Metadata for the vector, or None if not found
        """
        if index_name not in self.metadata:
            return None
            
        if vector_id >= len(self.metadata[index_name]):
            return None
            
        metadata_entry = self.metadata[index_name][vector_id]
        return {k: v for k, v in metadata_entry.items() if k != "text"}
    
    @trace_method("vector_store.list_indexes")
    def list_indexes(self) -> List[Dict[str, Any]]:
        """
        List all available indexes
        
        Returns:
            List of index information, including name and vector count
        """
        result = []
        for index_name, index in self.indexes.items():
            result.append({
                "name": index_name,
                "vector_count": index.ntotal,
                "metadata_count": len(self.metadata.get(index_name, []))
            })
        return result


# Create a singleton instance
vector_store = VectorStore()
