"""
Thread Persistence for OpenAI Assistants API

This module handles persisting thread IDs to maintain conversation state
across multiple requests and sessions.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Any
import asyncio

from ..config import settings
from ..monitoring.tracing import trace_method, trace_async_method

logger = logging.getLogger(__name__)

# Thread storage location
THREAD_STORAGE_DIR = os.path.join(os.path.dirname(__file__), '../../data/threads')


class ThreadPersistence:
    """
    Handles persisting thread IDs for OpenAI Assistants API
    
    This allows maintaining conversation state across multiple requests
    by reusing existing threads when appropriate.
    """
    
    def __init__(self):
        """Initialize thread persistence"""
        # Create thread storage directory if it doesn't exist
        os.makedirs(THREAD_STORAGE_DIR, exist_ok=True)
        
        # In-memory cache for thread IDs for faster access
        self._thread_cache: Dict[str, Dict[str, Any]] = {}
        
        # Load existing threads on startup
        self._load_threads()
    
    def _load_threads(self):
        """Load existing thread records from disk"""
        try:
            file_count = 0
            for filename in os.listdir(THREAD_STORAGE_DIR):
                if filename.endswith('.json'):
                    file_path = os.path.join(THREAD_STORAGE_DIR, filename)
                    with open(file_path, 'r') as f:
                        thread_data = json.load(f)
                        
                        # Index by key identifiers
                        if "dealer_id" in thread_data and "thread_id" in thread_data:
                            key = self._make_thread_key(
                                thread_data["dealer_id"], 
                                thread_data.get("conversation_type", "general")
                            )
                            self._thread_cache[key] = thread_data
                            file_count += 1
                            
            logger.info(f"Loaded {file_count} thread records from storage")
            
        except Exception as e:
            logger.error(f"Error loading thread records: {str(e)}")
    
    def _make_thread_key(self, dealer_id: str, conversation_type: str) -> str:
        """Create a unique key for thread lookup"""
        return f"{dealer_id}:{conversation_type}"
    
    @trace_method("thread_persistence.store_thread")
    def store_thread(
        self, 
        dealer_id: str, 
        thread_id: str, 
        conversation_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a thread ID for future use
        
        Args:
            dealer_id: Dealer ID
            thread_id: OpenAI thread ID
            conversation_type: Type of conversation (e.g., "inventory", "market")
            metadata: Optional additional information about the thread
            
        Returns:
            True if successful
        """
        try:
            key = self._make_thread_key(dealer_id, conversation_type)
            
            # Create thread record
            thread_data = {
                "dealer_id": dealer_id,
                "thread_id": thread_id,
                "conversation_type": conversation_type,
                "created_at": time.time(),
                "last_updated": time.time(),
                "metadata": metadata or {}
            }
            
            # Store in memory
            self._thread_cache[key] = thread_data
            
            # Save to disk
            self._persist_thread(thread_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing thread ID: {str(e)}")
            return False
    
    def _persist_thread(self, thread_data: Dict[str, Any]) -> None:
        """Save thread data to disk"""
        try:
            # Create filename based on dealer and thread ID
            filename = f"{thread_data['dealer_id']}_{thread_data['conversation_type']}.json"
            file_path = os.path.join(THREAD_STORAGE_DIR, filename)
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(thread_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error persisting thread data: {str(e)}")
    
    @trace_method("thread_persistence.get_thread")
    def get_thread(
        self, 
        dealer_id: str, 
        conversation_type: str = "general"
    ) -> Optional[str]:
        """
        Retrieve a stored thread ID
        
        Args:
            dealer_id: Dealer ID
            conversation_type: Type of conversation
            
        Returns:
            Thread ID if found, None otherwise
        """
        key = self._make_thread_key(dealer_id, conversation_type)
        
        if key in self._thread_cache:
            # Update last accessed time
            self._thread_cache[key]["last_accessed"] = time.time()
            return self._thread_cache[key]["thread_id"]
            
        return None
    
    @trace_method("thread_persistence.update_thread_metadata")
    def update_thread_metadata(
        self,
        dealer_id: str,
        conversation_type: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for a thread
        
        Args:
            dealer_id: Dealer ID
            conversation_type: Type of conversation
            metadata: New metadata to merge with existing
            
        Returns:
            True if successful
        """
        key = self._make_thread_key(dealer_id, conversation_type)
        
        if key in self._thread_cache:
            # Merge new metadata with existing
            self._thread_cache[key]["metadata"].update(metadata)
            self._thread_cache[key]["last_updated"] = time.time()
            
            # Persist changes
            self._persist_thread(self._thread_cache[key])
            return True
            
        return False
    
    @trace_method("thread_persistence.list_threads")
    def list_threads(self, dealer_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List stored threads
        
        Args:
            dealer_id: Optional dealer ID to filter by
            
        Returns:
            List of thread records
        """
        result = []
        
        for key, thread_data in self._thread_cache.items():
            if dealer_id is None or thread_data["dealer_id"] == dealer_id:
                result.append(thread_data)
                
        return result


# Create singleton instance
thread_persistence = ThreadPersistence()
