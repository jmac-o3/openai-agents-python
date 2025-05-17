"""
File Upload and OpenAI Vector Store Management for DOSO AI

This module provides capabilities to upload files to OpenAI's file API and
associate them with OpenAI Assistants for vector search and retrieval,
replacing the previous pgvector implementation.
"""

import json
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI
from openai.types.beta import Assistant
from openai.types.beta.threads import ThreadMessage, Run

from ..config import settings
from ..monitoring.tracing import trace_async_method, trace_method

logger = logging.getLogger(__name__)

class OpenAIFileStore:
    """
    File and vector storage implementation using OpenAI's file API and assistants
    """
    
    def __init__(self):
        """Initialize the OpenAI client with API key"""
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.assistant = None
        self.assistant_id = None
        
    @trace_method("openai.ensure_assistant_exists")
    def ensure_assistant_exists(self) -> str:
        """
        Ensure the DOSO AI assistant exists, creating it if necessary
        
        Returns:
            Assistant ID
        """
        # If we already have the assistant ID cached, return it
        if self.assistant_id:
            return self.assistant_id
            
        # Check for existing assistant with our name
        assistants = self.client.beta.assistants.list(
            order="desc",
            limit=100
        )
        
        for assistant in assistants.data:
            if assistant.name == "DOSO AI Assistant":
                self.assistant_id = assistant.id
                self.assistant = assistant
                logger.info(f"Found existing assistant: {self.assistant_id}")
                return self.assistant_id
        
        # Create new assistant if not found
        logger.info("Creating new DOSO AI Assistant")
        self.assistant = self.client.beta.assistants.create(
            name="DOSO AI Assistant",
            description="Ford Dealer Inventory Optimization System Assistant",
            model=settings.OPENAI_MODEL,
            tools=[
                {"type": "file_search"},
                {"type": "retrieval"}
            ],
            instructions="""
            You are the DOSO AI Assistant, designed to help Ford dealerships optimize inventory, 
            track allocations, analyze market trends, and plan orders effectively.
            
            Your responsibilities include:
            1. Analyzing dealership data from uploaded files (CSV, PDF, TXT)
            2. Providing inventory analysis and optimization recommendations
            3. Tracking vehicle allocations and order constraints
            4. Analyzing market trends and competitive positioning
            5. Identifying gaps in inventory mix and suggesting improvements
            6. Monitoring sales velocity and turnover metrics
            
            When analyzing data:
            - Extract relevant information from uploaded files using file_search
            - Maintain structured format for numerical data and metrics
            - Provide actionable insights specific to the dealer's situation
            - Reference specific models, time periods, and metrics in recommendations
            
            Ensure all recommendations are data-driven and specific to Ford dealerships.
            """
        )
        
        self.assistant_id = self.assistant.id
        logger.info(f"Created new assistant: {self.assistant_id}")
        return self.assistant_id
        
    @trace_method("openai.upload_file")
    def upload_file(self, file_path: str, purpose: str = "assistants") -> str:
        """
        Upload a file to OpenAI
        
        Args:
            file_path: Path to the file to upload
            purpose: File purpose (assistants or fine-tune)
            
        Returns:
            OpenAI file ID
        """
        try:
            with open(file_path, "rb") as file:
                response = self.client.files.create(
                    file=file,
                    purpose=purpose
                )
            logger.info(f"Uploaded file {file_path} to OpenAI with ID: {response.id}")
            return response.id
        except Exception as e:
            logger.error(f"Error uploading file {file_path}: {str(e)}")
            raise
            
    @trace_method("openai.upload_and_attach_file")
    def upload_and_attach_file(self, file_path: str) -> str:
        """
        Upload a file and attach it to the assistant
        
        Args:
            file_path: Path to the file to upload
            
        Returns:
            OpenAI file ID
        """
        # Ensure assistant exists
        assistant_id = self.ensure_assistant_exists()
        
        # Upload file
        file_id = self.upload_file(file_path)
        
        # Attach file to the assistant
        self.client.beta.assistants.files.create(
            assistant_id=assistant_id,
            file_id=file_id
        )
        
        logger.info(f"Attached file {file_id} to assistant {assistant_id}")
        return file_id
        
    @trace_method("openai.list_assistant_files")
    def list_assistant_files(self) -> List[Dict[str, Any]]:
        """
        List all files attached to the assistant
        
        Returns:
            List of file information dictionaries
        """
        # Ensure assistant exists
        assistant_id = self.ensure_assistant_exists()
        
        # Get files
        files = self.client.beta.assistants.files.list(
            assistant_id=assistant_id
        )
        
        # Convert to list of dicts
        return [
            {
                "id": file.id,
                "object": file.object,
                "created_at": file.created_at,
                "assistant_id": file.assistant_id
            }
            for file in files.data
        ]
    
    @trace_method("openai.detach_file")
    def detach_file(self, file_id: str) -> bool:
        """
        Detach a file from the assistant
        
        Args:
            file_id: OpenAI file ID
            
        Returns:
            True if detached successfully
        """
        # Ensure assistant exists
        assistant_id = self.ensure_assistant_exists()
        
        try:
            # Delete the file attachment (not the file itself)
            self.client.beta.assistants.files.delete(
                assistant_id=assistant_id,
                file_id=file_id
            )
            logger.info(f"Detached file {file_id} from assistant {assistant_id}")
            return True
        except Exception as e:
            logger.error(f"Error detaching file {file_id}: {str(e)}")
            return False
    
    @trace_method("openai.delete_file")
    def delete_file(self, file_id: str) -> bool:
        """
        Delete a file from OpenAI
        
        Args:
            file_id: OpenAI file ID
            
        Returns:
            True if deleted successfully
        """
        try:
            # Detach file first if attached
            self.detach_file(file_id)
            
            # Delete the file
            self.client.files.delete(file_id)
            logger.info(f"Deleted file {file_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file {file_id}: {str(e)}")
            return False
    
    @trace_async_method("openai.create_thread_and_run")
    async def create_thread_and_run(
        self, 
        query: str, 
        file_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Create a thread and run with the assistant
        
        Args:
            query: User query to assistant
            file_ids: Optional additional file IDs to include for this thread
            metadata: Optional metadata for the thread
            
        Returns:
            Dict with thread_id, run_id and final response
        """
        # Ensure assistant exists
        assistant_id = self.ensure_assistant_exists()
        
        # Create thread with optional file attachments
        thread = self.client.beta.threads.create(
            messages=[
                {
                    "role": "user",
                    "content": query,
                    "file_ids": file_ids or []
                }
            ],
            metadata=metadata or {}
        )
        
        # Start the run
        run = self.client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )
        
        # Poll for completion
        run = self._wait_for_run_completion(thread.id, run.id)
        
        # Get the assistant's response
        messages = self.client.beta.threads.messages.list(
            thread_id=thread.id
        )
        
        # Extract the response content
        assistant_responses = [
            msg for msg in messages.data 
            if msg.role == "assistant"
        ]
        
        if assistant_responses:
            response_texts = []
            for content_item in assistant_responses[0].content:
                if content_item.type == "text":
                    response_texts.append(content_item.text.value)
            
            response_text = "\n".join(response_texts)
        else:
            response_text = "No response from assistant"
        
        return {
            "thread_id": thread.id,
            "run_id": run.id,
            "status": run.status,
            "response": response_text
        }
    
    def _wait_for_run_completion(self, thread_id: str, run_id: str) -> Any:
        """
        Poll for run completion
        
        Args:
            thread_id: Thread ID
            run_id: Run ID
            
        Returns:
            Completed run object
        """
        while True:
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run_id
            )
            
            if run.status in ["completed", "failed", "cancelled", "expired"]:
                return run
                
            # Wait before polling again
            import time
            time.sleep(1)
    
    @trace_async_method("openai.run_file_search_query")
    async def run_file_search_query(
        self, 
        query: str, 
        metadata_filter: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Run a semantic search query against uploaded files
        
        Args:
            query: Search query
            metadata_filter: Optional metadata filtering
            
        Returns:
            List of search results
        """
        # Construct a specialized prompt for file search
        search_prompt = f"""
        Please search through all available files for information about: {query}
        
        Use your file_search tool to find the most relevant information.
        Only return information that's relevant to the query.
        Format the results as a JSON list of objects with the following structure:
        
        [
            {{
                "source": "file name",
                "content": "relevant content",
                "relevance": "why this is relevant"
            }}
        ]
        
        Make sure to use proper JSON formatting.
        """
        
        # Run the query with optional metadata filtering
        result = await self.create_thread_and_run(
            query=search_prompt,
            metadata=metadata_filter
        )
        
        # Try to parse the response as JSON
        try:
            response_text = result["response"]
            
            # Extract JSON from response if needed (handle markdown code blocks)
            if "```json" in response_text:
                import re
                json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1)
            
            search_results = json.loads(response_text)
            return search_results
        except json.JSONDecodeError:
            logger.error("Failed to parse response as JSON")
            return [{
                "source": "error",
                "content": result["response"],
                "relevance": "Error parsing response as JSON"
            }]


# Create singleton instance
file_store = OpenAIFileStore()
