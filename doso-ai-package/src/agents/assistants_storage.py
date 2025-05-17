"""
Assistants-based storage adapter for DOSO AI using openai-agents SDK

This module replaces PostgreSQL/Redis storage with openai-agents' built-in
support for OpenAI Assistants, using the existing Agent framework.

Features:
- Thread persistence for continuity across requests
- Rate limit handling with exponential backoff
- File batch monitoring for indexing completeness
"""

import logging
from typing import Dict, List, Any, Optional, Union
import tempfile
import os
import json
import asyncio

from openai_agents import Agent, RunConfig, run_agent, function_tool
from openai.types.beta.threads import ThreadCreateParams

from ..config import settings
from ..monitoring.tracing import trace_method, trace_async_method
from ..utils.thread_persistence import thread_persistence
from ..utils.rate_limit_handler import with_async_retry
from ..utils.file_batch_monitor import file_batch_monitor

logger = logging.getLogger(__name__)


class AssistantsStorageAgent(Agent):
    """
    Agent that handles file storage and retrieval using OpenAI Assistants
    instead of PostgreSQL/Redis. This leverages openai-agents SDK's built-in
    support for OpenAI Assistants API.
    
    Features:
    - Thread persistence for continuity across requests
    - Rate limit handling with exponential backoff
    - File batch monitoring for indexing completeness
    """
    
    def __init__(self):
        """Initialize the Assistants Storage agent"""
        super().__init__(
            name="Assistants Storage Agent",
            instructions="""
            You manage file storage and retrieval for DOSO AI, handling:
            
            1. Semantic search across uploaded documents
            2. File storage and organization
            3. Data persistence across sessions
            4. Context maintenance for multi-step analyses
            
            Use your file_search and retrieval tools to find relevant information
            in uploaded documents. Return results in structured formats requested
            by your tools.
            """,
            model=settings.OPENAI_MODEL,
            tools=["file_search", "retrieval"]
        )
        
        # Track current active threads
        self._active_threads: Dict[str, str] = {}
    
    @function_tool
    @with_async_retry
    async def search_documents(
        self, 
        query: str,
        dealer_id: Optional[str] = None,
        conversation_type: str = "search"
    ) -> List[Dict[str, Any]]:
        """
        Search uploaded documents for specific information
        
        Args:
            query: Search query to look for in documents
            dealer_id: Optional dealer ID for thread persistence
            conversation_type: Type of conversation for thread tracking
        
        Returns:
            List of search results with source, content and relevance score
        """
        # This will use OpenAI file_search tool under the hood through openai-agents
        results = []
        
        # Use persistent thread if dealer_id is provided
        thread_id = None
        if dealer_id:
            thread_id = thread_persistence.get_thread(dealer_id, conversation_type)
            
            # Register a new thread for next time if we don't have one
            thread_metadata = {
                "query": query,
                "conversation_type": conversation_type,
                "timestamp": "current_timestamp"
            }
            
            if thread_id:
                # Update existing thread metadata
                thread_persistence.update_thread_metadata(
                    dealer_id, 
                    conversation_type,
                    thread_metadata
                )
        
        # Format results in a consistent way
        try:
            # Create run options with thread_id if available
            run_options = {}
            if thread_id:
                run_options["thread_id"] = thread_id
                
            response = await self.ask(
                f"""
                Search through all available files for information about: {query}
                
                Use your file_search tool to find the most relevant information.
                Format the results as a JSON list of objects with the following structure:
                
                [
                    {{
                        "source": "file name",
                        "content": "relevant content",
                        "relevance": "why this is relevant"
                    }}
                ]
                
                Make sure to use proper JSON formatting. Return ONLY the JSON.
                """,
                **run_options
            )
            
            # Store thread ID for future use if dealer_id was provided and a new thread was created
            if dealer_id and hasattr(response, "thread_id") and response.thread_id:
                new_thread_id = response.thread_id
                if not thread_id:  # Only store if this is a new thread
                    thread_persistence.store_thread(
                        dealer_id=dealer_id,
                        thread_id=new_thread_id,
                        conversation_type=conversation_type,
                        metadata={
                            "query": query,
                            "conversation_type": conversation_type,
                            "created_at": "current_timestamp"
                        }
                    )
                # Track in memory
                self._active_threads[f"{dealer_id}:{conversation_type}"] = new_thread_id
            
            # Extract JSON if it's wrapped in markdown code blocks
            content = response.content
            if "```json" in content:
                import re
                json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
            elif "```" in content:
                import re
                json_match = re.search(r'```\n(.*?)\n```', content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
                    
            # Parse the JSON response
            results = json.loads(content)
            return results
        
        except Exception as e:
            logger.error(f"Error searching documents: {str(e)}")
            return [{
                "source": "error",
                "content": str(e),
                "relevance": "Error occurred during search"
            }]
    
    @function_tool
    @with_async_retry
    async def store_analysis_result(
        self, 
        dealer_id: str, 
        analysis_type: str, 
        result_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Store analysis results for future reference
        
        Args:
            dealer_id: Dealer identifier
            analysis_type: Type of analysis (inventory, market, etc.)
            result_data: Analysis results to store
            
        Returns:
            Status information about the storage operation
        """
        # Create a temporary file to store the results
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp:
                # Create metadata about this result
                metadata = {
                    "dealer_id": dealer_id,
                    "analysis_type": analysis_type,
                    "timestamp": result_data.get("timestamp", "unknown")
                }
                
                # Write data with metadata
                json.dump({
                    "metadata": metadata,
                    "result": result_data
                }, temp, default=str, indent=2)
                
                temp_path = temp.name
                
            # Get assistant_id from the OpenAI Agent instance
            assistant_id = None
            if hasattr(self, "openai_assistant_id"):
                assistant_id = self.openai_assistant_id
            
            file_ids = []
            batch_id = None
            
            # Create file in OpenAI and attach to assistant
            from openai import OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Upload the file to OpenAI
            try:
                with open(temp_path, "rb") as file:
                    response = client.files.create(
                        file=file,
                        purpose="assistants"
                    )
                    file_id = response.id
                    file_ids.append(file_id)
                    
                # If we have assistant_id, attach the file and monitor its indexing
                if assistant_id and file_id:
                    # Attach the file to the assistant
                    client.beta.assistants.files.create(
                        assistant_id=assistant_id,
                        file_id=file_id
                    )
                    
                    # Register with file batch monitor
                    batch_id = file_batch_monitor.register_batch(
                        file_ids=[file_id],
                        assistant_id=assistant_id,
                        metadata={
                            "dealer_id": dealer_id,
                            "analysis_type": analysis_type
                        }
                    )
                    
                    # Start monitoring the batch in the background
                    asyncio.create_task(file_batch_monitor.wait_for_batch(batch_id))
                    
            except Exception as e:
                logger.error(f"Error uploading file to OpenAI: {str(e)}")
                raise
            
            # Clean up the temp file
            try:
                os.unlink(temp_path)
            except:
                pass
                
            return {
                "status": "success",
                "dealer_id": dealer_id,
                "analysis_type": analysis_type,
                "message": "Analysis results stored successfully",
                "file_ids": file_ids,
                "batch_id": batch_id
            }
            
        except Exception as e:
            logger.error(f"Error storing analysis results: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to store results: {str(e)}"
            }
    
    @function_tool
    @with_async_retry
    async def retrieve_previous_analysis(
        self, 
        dealer_id: str, 
        analysis_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve previous analysis results for a dealer
        
        Args:
            dealer_id: Dealer identifier
            analysis_type: Optional type of analysis to filter by
            
        Returns:
            List of previous analysis results matching criteria
        """
        # Get thread ID for dealer if available (for context continuity)
        thread_id = thread_persistence.get_thread(
            dealer_id, 
            f"analysis_{analysis_type}" if analysis_type else "analysis"
        )
        
        # Construct a search query based on parameters
        query = f"dealer_id: {dealer_id}"
        if analysis_type:
            query += f" analysis_type: {analysis_type}"
            
        # Use the search_documents method to find relevant results
        try:
            results = await self.search_documents(
                query=query,
                dealer_id=dealer_id,
                conversation_type=f"analysis_{analysis_type}" if analysis_type else "analysis"
            )
            
            # Filter and process results to match expected format
            processed_results = []
            for result in results:
                # Extract content and parse if it's JSON
                content = result.get("content", "{}")
                if isinstance(content, str):
                    try:
                        content_data = json.loads(content)
                        processed_results.append(content_data)
                    except:
                        # If not valid JSON, just add as is
                        processed_results.append({"raw_content": content})
                else:
                    processed_results.append(content)
                        
            return processed_results
            
        except Exception as e:
            logger.error(f"Error retrieving previous analysis: {str(e)}")
            return [{
                "status": "error",
                "message": f"Failed to retrieve previous analysis: {str(e)}"
            }]
    
    @function_tool
    @with_async_retry
    async def upload_file(
        self,
        file_path: str,
        dealer_id: str,
        file_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Upload a file to the Assistant for processing
        
        Args:
            file_path: Path to the file to upload
            dealer_id: Dealer identifier
            file_type: Type of file (inventory, market, etc.)
            
        Returns:
            Status information about the upload
        """
        # Verify the file exists
        if not os.path.exists(file_path):
            return {
                "status": "error",
                "message": f"File not found: {file_path}"
            }
            
        # Get assistant_id from the OpenAI Agent instance
        assistant_id = None
        if hasattr(self, "openai_assistant_id"):
            assistant_id = self.openai_assistant_id
        else:
            return {
                "status": "error", 
                "message": "Assistant ID not available"
            }
            
        # Upload the file to OpenAI
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
            with open(file_path, "rb") as file:
                response = client.files.create(
                    file=file,
                    purpose="assistants"
                )
                file_id = response.id
                
            # Attach the file to the assistant
            client.beta.assistants.files.create(
                assistant_id=assistant_id,
                file_id=file_id
            )
            
            # Register with file batch monitor
            batch_id = file_batch_monitor.register_batch(
                file_ids=[file_id],
                assistant_id=assistant_id,
                metadata={
                    "dealer_id": dealer_id,
                    "file_type": file_type,
                    "original_path": file_path
                }
            )
            
            # Start monitoring in background
            asyncio.create_task(file_batch_monitor.wait_for_batch(batch_id))
            
            return {
                "status": "success",
                "message": "File uploaded successfully",
                "file_id": file_id,
                "batch_id": batch_id,
                "dealer_id": dealer_id
            }
            
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to upload file: {str(e)}"
            }
    
    @function_tool
    @with_async_retry
    async def check_file_batch_status(
        self,
        batch_id: str
    ) -> Dict[str, Any]:
        """
        Check the status of a file batch
        
        Args:
            batch_id: Batch ID to check
            
        Returns:
            Status information about the batch
        """
        return file_batch_monitor.get_batch_status(batch_id)
    
    @function_tool
    @with_async_retry
    async def create_conversation_thread(
        self,
        dealer_id: str,
        conversation_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a new conversation thread
        
        Args:
            dealer_id: Dealer identifier
            conversation_type: Type of conversation
            metadata: Optional metadata for the thread
            
        Returns:
            Status information about the thread creation
        """
        # Check if a thread already exists
        existing_thread_id = thread_persistence.get_thread(dealer_id, conversation_type)
        if existing_thread_id:
            # Update metadata if provided
            if metadata:
                thread_persistence.update_thread_metadata(
                    dealer_id,
                    conversation_type,
                    metadata
                )
                
            return {
                "status": "success",
                "message": "Using existing thread",
                "thread_id": existing_thread_id,
                "is_new": False
            }
            
        # Create a new thread
        try:
            from openai import OpenAI
            client = OpenAI(api_key=settings.OPENAI_API_KEY)
            
            # Create thread with metadata
            thread_params: ThreadCreateParams = {}
            if metadata:
                thread_params["metadata"] = metadata
                
            thread = client.beta.threads.create(**thread_params)
            thread_id = thread.id
            
            # Store in persistence
            thread_persistence.store_thread(
                dealer_id=dealer_id,
                thread_id=thread_id,
                conversation_type=conversation_type,
                metadata=metadata
            )
            
            # Track in memory
            self._active_threads[f"{dealer_id}:{conversation_type}"] = thread_id
            
            return {
                "status": "success",
                "message": "Thread created successfully",
                "thread_id": thread_id,
                "is_new": True
            }
            
        except Exception as e:
            logger.error(f"Error creating thread: {str(e)}")
            return {
                "status": "error",
                "message": f"Failed to create thread: {str(e)}"
            }


# Create a singleton instance
assistants_storage = AssistantsStorageAgent()
