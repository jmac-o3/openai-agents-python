"""
File Batch Monitoring for OpenAI Assistants API

This module handles monitoring the indexing status of file batches
for OpenAI Assistants API, ensuring files are fully processed before use.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Set, Tuple, Any

from openai import OpenAI
from openai.types.beta.assistants import FileDeleted

from ..config import settings
from ..monitoring.tracing import trace_method, trace_async_method
from .rate_limit_handler import with_retry, with_async_retry

logger = logging.getLogger(__name__)


class FileBatchMonitor:
    """
    Monitors the indexing status of file batches for OpenAI Assistants
    
    Ensures that files are fully processed and indexed before they are
    used in conversations, preventing issues with incomplete indexing.
    """
    
    def __init__(
        self,
        polling_interval: float = 2.0,
        max_attempts: int = 15,
        batch_timeout: float = 300.0
    ):
        """
        Initialize the file batch monitor
        
        Args:
            polling_interval: Time in seconds between polling attempts
            max_attempts: Maximum number of polling attempts
            batch_timeout: Maximum time in seconds to wait for a batch to complete
        """
        self.polling_interval = polling_interval
        self.max_attempts = max_attempts
        self.batch_timeout = batch_timeout
        
        # OpenAI client
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Tracking
        self._active_batches: Dict[str, Dict[str, Any]] = {}
        self._completed_batches: Set[str] = set()
        self._failed_batches: Dict[str, str] = {}
    
    @trace_method("file_batch.generate_batch_id")
    def generate_batch_id(self, file_ids: List[str], assistant_id: str) -> str:
        """
        Generate a batch ID for tracking a set of files
        
        Args:
            file_ids: List of file IDs in the batch
            assistant_id: Assistant ID associated with the batch
            
        Returns:
            Batch ID for tracking
        """
        # Sort file IDs for consistent batch IDs
        sorted_ids = sorted(file_ids)
        
        # Create a unique batch ID using timestamp and file count
        timestamp = int(time.time())
        batch_id = f"batch_{assistant_id}_{timestamp}_{len(sorted_ids)}"
        
        return batch_id
    
    @trace_method("file_batch.register_batch")
    def register_batch(
        self, 
        file_ids: List[str], 
        assistant_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register a batch of files for monitoring
        
        Args:
            file_ids: List of file IDs in the batch
            assistant_id: Assistant ID associated with the batch
            metadata: Optional metadata about the batch
            
        Returns:
            Batch ID for tracking
        """
        batch_id = self.generate_batch_id(file_ids, assistant_id)
        
        # Store batch information
        self._active_batches[batch_id] = {
            "file_ids": file_ids,
            "assistant_id": assistant_id,
            "registered_at": time.time(),
            "status": "registered",
            "metadata": metadata or {},
            "polling_attempts": 0,
        }
        
        logger.info(f"Registered file batch {batch_id} with {len(file_ids)} files")
        return batch_id
    
    @with_retry
    def _check_file_status(self, file_id: str, assistant_id: str) -> str:
        """
        Check the status of a single file
        
        Args:
            file_id: File ID to check
            assistant_id: Assistant ID the file is attached to
            
        Returns:
            Status of the file ('processed', 'processing', 'error', 'not_found')
        """
        try:
            # Get the file from the assistant
            assistant_file = self.client.beta.assistants.files.retrieve(
                assistant_id=assistant_id,
                file_id=file_id
            )
            
            # Return status based on file object
            if hasattr(assistant_file, "status") and assistant_file.status:
                return assistant_file.status
                
            # If no explicit status, assume processed
            return "processed"
            
        except FileDeleted:
            # File was deleted
            return "not_found"
            
        except Exception as e:
            logger.error(f"Error checking file status for {file_id}: {str(e)}")
            return "error"
    
    @with_async_retry
    async def _check_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Check the status of a batch of files
        
        Args:
            batch_id: Batch ID to check
            
        Returns:
            Status report for the batch
        """
        if batch_id not in self._active_batches:
            return {
                "batch_id": batch_id,
                "status": "unknown",
                "message": "Batch ID not found"
            }
            
        batch = self._active_batches[batch_id]
        file_ids = batch["file_ids"]
        assistant_id = batch["assistant_id"]
        
        # Track statuses for all files
        file_statuses: Dict[str, str] = {}
        processed_count = 0
        error_count = 0
        not_found_count = 0
        
        # Check each file
        for file_id in file_ids:
            status = self._check_file_status(file_id, assistant_id)
            file_statuses[file_id] = status
            
            if status == "processed":
                processed_count += 1
            elif status == "error":
                error_count += 1
            elif status == "not_found":
                not_found_count += 1
        
        # Update batch information
        batch["last_checked"] = time.time()
        batch["polling_attempts"] += 1
        batch["file_statuses"] = file_statuses
        
        # Determine overall batch status
        if processed_count + not_found_count == len(file_ids):
            # All files processed or not found
            batch["status"] = "completed"
            self._completed_batches.add(batch_id)
            self._active_batches.pop(batch_id, None)
            
            logger.info(f"Batch {batch_id} completed: {processed_count} processed, {not_found_count} not found")
            
        elif error_count > 0:
            # At least one file had an error
            batch["status"] = "error"
            error_message = f"Batch contains {error_count} files with errors"
            self._failed_batches[batch_id] = error_message
            self._active_batches.pop(batch_id, None)
            
            logger.error(f"Batch {batch_id} failed: {error_message}")
            
        else:
            # Still processing
            batch["status"] = "processing"
            logger.info(f"Batch {batch_id} processing: {processed_count}/{len(file_ids)} complete")
        
        # Check for timeout
        if batch["status"] == "processing":
            elapsed = time.time() - batch["registered_at"]
            if elapsed > self.batch_timeout:
                batch["status"] = "timeout"
                error_message = f"Batch processing timed out after {elapsed:.1f}s"
                self._failed_batches[batch_id] = error_message
                self._active_batches.pop(batch_id, None)
                
                logger.error(f"Batch {batch_id} failed: {error_message}")
        
        return {
            "batch_id": batch_id,
            "status": batch.get("status", "unknown"),
            "processed_count": processed_count,
            "total_count": len(file_ids),
            "error_count": error_count,
            "not_found_count": not_found_count,
            "polling_attempts": batch["polling_attempts"],
            "file_statuses": file_statuses
        }
    
    @trace_async_method("file_batch.wait_for_batch")
    async def wait_for_batch(self, batch_id: str) -> Dict[str, Any]:
        """
        Wait for a batch of files to complete processing
        
        Args:
            batch_id: Batch ID to wait for
            
        Returns:
            Final status of the batch
        """
        if batch_id not in self._active_batches:
            # Check if this was a previously completed or failed batch
            if batch_id in self._completed_batches:
                return {
                    "batch_id": batch_id,
                    "status": "completed",
                    "message": "Batch was previously completed"
                }
            elif batch_id in self._failed_batches:
                return {
                    "batch_id": batch_id,
                    "status": "error",
                    "message": self._failed_batches[batch_id]
                }
            else:
                return {
                    "batch_id": batch_id,
                    "status": "unknown",
                    "message": "Batch ID not found"
                }
        
        # Poll until completion, error, or timeout
        attempt = 0
        status_report = None
        
        while attempt < self.max_attempts:
            status_report = await self._check_batch_status(batch_id)
            
            if status_report["status"] in ["completed", "error", "timeout"]:
                # Batch is done processing
                return status_report
            
            # Wait before checking again
            await asyncio.sleep(self.polling_interval)
            attempt += 1
        
        # If we reach here, we've exceeded max attempts
        if batch_id in self._active_batches:
            self._active_batches[batch_id]["status"] = "timeout"
            error_message = f"Exceeded maximum polling attempts ({self.max_attempts})"
            self._failed_batches[batch_id] = error_message
            self._active_batches.pop(batch_id, None)
            
            logger.error(f"Batch {batch_id} failed: {error_message}")
        
        return {
            "batch_id": batch_id,
            "status": "timeout",
            "message": f"Exceeded maximum polling attempts ({self.max_attempts})",
            "last_status": status_report
        }
    
    @trace_method("file_batch.get_batch_status")
    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get the current status of a batch without waiting
        
        Args:
            batch_id: Batch ID to check
            
        Returns:
            Current status information
        """
        if batch_id in self._active_batches:
            batch = self._active_batches[batch_id]
            return {
                "batch_id": batch_id,
                "status": batch.get("status", "unknown"),
                "registered_at": batch["registered_at"],
                "file_count": len(batch["file_ids"]),
                "polling_attempts": batch["polling_attempts"],
                "file_statuses": batch.get("file_statuses", {})
            }
        elif batch_id in self._completed_batches:
            return {
                "batch_id": batch_id,
                "status": "completed",
                "message": "Batch was previously completed"
            }
        elif batch_id in self._failed_batches:
            return {
                "batch_id": batch_id,
                "status": "error",
                "message": self._failed_batches[batch_id]
            }
        else:
            return {
                "batch_id": batch_id,
                "status": "unknown",
                "message": "Batch ID not found"
            }
    
    @trace_method("file_batch.list_batches")
    def list_batches(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all batches, optionally filtered by status
        
        Args:
            status: Optional status to filter by ('active', 'completed', 'failed')
            
        Returns:
            List of batch information
        """
        result = []
        
        # Add active batches
        if status is None or status == "active":
            for batch_id, batch in self._active_batches.items():
                result.append({
                    "batch_id": batch_id,
                    "status": batch.get("status", "unknown"),
                    "registered_at": batch["registered_at"],
                    "file_count": len(batch["file_ids"]),
                    "polling_attempts": batch["polling_attempts"]
                })
        
        # Add completed batches
        if status is None or status == "completed":
            for batch_id in self._completed_batches:
                result.append({
                    "batch_id": batch_id,
                    "status": "completed"
                })
        
        # Add failed batches
        if status is None or status == "failed":
            for batch_id, error_message in self._failed_batches.items():
                result.append({
                    "batch_id": batch_id,
                    "status": "error",
                    "message": error_message
                })
        
        return result


# Create singleton instance
file_batch_monitor = FileBatchMonitor()
