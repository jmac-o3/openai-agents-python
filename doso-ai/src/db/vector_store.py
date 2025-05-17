"""
Vector Storage for semantic search capabilities in DOSO AI

This module provides a vector database implementation for storing and 
retrieving document embeddings for semantic search functionality.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from openai import OpenAI
from sqlalchemy import Column, String, Table, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from ..config import settings
from ..db.dependencies import AsyncSessionLocal

logger = logging.getLogger(__name__)

# Embedding model configuration
EMBEDDING_MODEL = "text-embedding-ada-002"  # Default OpenAI embedding model
EMBEDDING_DIMENSION = 1536  # Dimension for OpenAI ada-002 embeddings

# Vector distance threshold for semantic search
SIMILARITY_THRESHOLD = 0.75


class VectorStore:
    """
    Vector storage implementation using pgvector for semantic search
    """
    
    def __init__(self):
        """Initialize the vector store with OpenAI client and DB connection"""
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.table_name = "document_embeddings"

    async def ensure_schema_exists(self) -> None:
        """
        Ensure the required database schema and extensions exist
        """
        async with AsyncSessionLocal() as session:
            # Check if pgvector extension is installed
            await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            # Create table if not exists
            await session.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id TEXT PRIMARY KEY,
                content JSONB,
                metadata JSONB,
                embedding vector({EMBEDDING_DIMENSION})
            );
            """))
            
            # Create index for vector search
            await session.execute(text(f"""
            CREATE INDEX IF NOT EXISTS embedding_idx 
            ON {self.table_name} 
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """))
            
            await session.commit()

    async def add_document(
        self, 
        document_id: str, 
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a document to the vector store
        
        Args:
            document_id: Unique identifier for the document
            content: Document content as a dictionary
            metadata: Optional metadata for the document
        """
        # Generate the embedding for the document
        embedding = await self._generate_embedding(json.dumps(content))
        
        # Default empty metadata
        if metadata is None:
            metadata = {}
        
        async with AsyncSessionLocal() as session:
            # Insert or update document
            await session.execute(text(f"""
            INSERT INTO {self.table_name} (id, content, metadata, embedding)
            VALUES (:id, :content, :metadata, :embedding)
            ON CONFLICT (id) DO UPDATE
            SET content = :content,
                metadata = :metadata,
                embedding = :embedding;
            """), {
                "id": document_id,
                "content": json.dumps(content),
                "metadata": json.dumps(metadata),
                "embedding": embedding
            })
            
            await session.commit()

    async def search_similar(
        self,
        query: str,
        limit: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query
        
        Args:
            query: Search query string
            limit: Maximum number of results to return
            metadata_filter: Optional filter for metadata fields
            
        Returns:
            List of documents with similarity scores
        """
        # Generate embedding for the query
        query_embedding = await self._generate_embedding(query)
        
        # Build SQL query
        sql = f"""
        SELECT id, content, metadata, 
               1 - (embedding <=> :embedding) as similarity
        FROM {self.table_name}
        """
        
        # Add metadata filtering if needed
        params = {"embedding": query_embedding}
        where_clauses = []
        
        if metadata_filter:
            for key, value in metadata_filter.items():
                where_clauses.append(f"metadata->>'%s' = :%s" % (key, key))
                params[key] = value
                
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
            
        # Add similarity threshold and ordering
        sql += f"""
        AND 1 - (embedding <=> :embedding) > {SIMILARITY_THRESHOLD}
        ORDER BY similarity DESC
        LIMIT :limit
        """
        params["limit"] = limit
        
        # Execute search
        async with AsyncSessionLocal() as session:
            result = await session.execute(text(sql), params)
            rows = result.mappings().all()
            
            documents = []
            for row in rows:
                documents.append({
                    "id": row["id"],
                    "content": json.loads(row["content"]),
                    "metadata": json.loads(row["metadata"]),
                    "similarity": float(row["similarity"])
                })
            
            return documents

    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document from the vector store
        
        Args:
            document_id: ID of document to delete
            
        Returns:
            True if document was deleted, False if not found
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text(f"DELETE FROM {self.table_name} WHERE id = :id"),
                {"id": document_id}
            )
            
            await session.commit()
            
            # Check if any rows were deleted
            return result.rowcount > 0

    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text using OpenAI API
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zero vector as fallback
            return [0.0] * EMBEDDING_DIMENSION

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by ID
        
        Args:
            document_id: ID of document to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                text(f"SELECT id, content, metadata FROM {self.table_name} WHERE id = :id"),
                {"id": document_id}
            )
            
            row = result.mappings().first()
            
            if row:
                return {
                    "id": row["id"],
                    "content": json.loads(row["content"]),
                    "metadata": json.loads(row["metadata"]),
                }
            
            return None

    async def bulk_add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> None:
        """
        Add multiple documents to the vector store in a single transaction
        
        Args:
            documents: List of documents to add, each with id, content, and metadata
        """
        async with AsyncSessionLocal() as session:
            for doc in documents:
                document_id = doc["id"]
                content = doc["content"]
                metadata = doc.get("metadata", {})
                
                # Generate embedding
                embedding = await self._generate_embedding(json.dumps(content))
                
                # Add to session
                await session.execute(text(f"""
                INSERT INTO {self.table_name} (id, content, metadata, embedding)
                VALUES (:id, :content, :metadata, :embedding)
                ON CONFLICT (id) DO UPDATE
                SET content = :content,
                    metadata = :metadata,
                    embedding = :embedding;
                """), {
                    "id": document_id,
                    "content": json.dumps(content),
                    "metadata": json.dumps(metadata),
                    "embedding": embedding
                })
                
            # Commit all changes
            await session.commit()
            
    async def search_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for documents matching metadata criteria
        
        Args:
            metadata_filter: Filter criteria for document metadata
            limit: Maximum number of documents to return
            
        Returns:
            List of matching documents
        """
        sql = f"SELECT id, content, metadata FROM {self.table_name}"
        params = {"limit": limit}
        
        where_clauses = []
        for key, value in metadata_filter.items():
            where_clauses.append(f"metadata->>'%s' = :%s" % (key, key))
            params[key] = value
        
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
            
        sql += " LIMIT :limit"
        
        async with AsyncSessionLocal() as session:
            result = await session.execute(text(sql), params)
            rows = result.mappings().all()
            
            documents = []
            for row in rows:
                documents.append({
                    "id": row["id"],
                    "content": json.loads(row["content"]),
                    "metadata": json.loads(row["metadata"]),
                })
            
            return documents
