"""
FeedbackCollectorAgent - Processes sales outcome data and builds vector search index

This agent processes feedback data from CSV files containing sales outcomes,
validates them, stores them in a persistent JSONL store, and builds a FAISS
vector index for pattern discovery and learning.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator
import faiss

from agents import Agent, RunContextWrapper, function_tool
from openai import OpenAI

# Ensure the data directory exists
DATA_DIR = Path("doso-ai/data")
PERFORMANCE_LOG = DATA_DIR / "performance_log.jsonl"
VECTOR_INDEX_PATH = DATA_DIR / "vector_store" / "feedback_index.faiss"
VECTOR_METADATA_PATH = DATA_DIR / "vector_store" / "feedback_metadata.json"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATA_DIR / "vector_store", exist_ok=True)


class FeedbackRecord(BaseModel):
    """Feedback record from sales outcomes"""
    config_id: str
    sale_date: str  # ISO format date
    gross_profit: float
    ddt: int  # days to turn
    recommended_qty: int
    actual_sold: int
    constraint_hit: bool = False
    forecast_accuracy: Optional[float] = None
    outcome_rating: Optional[float] = None
    embedding: Optional[List[float]] = None
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    @field_validator('sale_date')
    @classmethod
    def validate_date(cls, v):
        """Validate date format"""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            try:
                # Try to parse as MM/DD/YYYY
                dt = datetime.strptime(v, '%m/%d/%Y')
                return dt.isoformat()
            except ValueError:
                try:
                    # Try to parse as YYYY-MM-DD
                    dt = datetime.strptime(v, '%Y-%m-%d')
                    return dt.isoformat()
                except ValueError:
                    raise ValueError(f"Invalid date format: {v}")


class FeedbackSummary(BaseModel):
    """Summary statistics for feedback records"""
    total_records: int
    successful_records: int
    failed_records: int = 0
    avg_gross_profit: float
    avg_ddt: float
    avg_outcome_rating: Optional[float] = None
    vectors_created: int = 0
    status: str = "success"
    message: Optional[str] = None


class FeedbackQuery(BaseModel):
    """Query parameters for searching feedback data"""
    query_text: str
    limit: int = 5
    min_score: float = 0.7


class FeedbackSearchResult(TypedDict):
    """Result of a semantic search in the feedback data"""
    record_id: str
    config_id: str
    sale_date: str
    text: str
    score: float


class VectorMetadata(TypedDict):
    """Metadata for the vector store"""
    record_ids: List[str]
    config_ids: List[str]
    sale_dates: List[str]
    dimensions: int
    total_records: int
    last_updated: str


def create_embedding_client():
    """Create an OpenAI client for embeddings"""
    return OpenAI()


async def create_embedding(text: str) -> List[float]:
    """Create an embedding from text using OpenAI's API"""
    client = create_embedding_client()
    
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        # Return a random embedding for demo purposes if OpenAI API fails
        return [0.0] * 3072


def save_feedback_record(record: FeedbackRecord) -> bool:
    """Save a feedback record to the JSONL store"""
    with open(PERFORMANCE_LOG, "a") as f:
        f.write(json.dumps(record.model_dump()) + "\n")
    return True


def load_all_feedback_records() -> List[FeedbackRecord]:
    """Load all feedback records from the JSONL store"""
    records = []
    
    if not PERFORMANCE_LOG.exists():
        return records
    
    with open(PERFORMANCE_LOG, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record_data = json.loads(line)
                records.append(FeedbackRecord.model_validate(record_data))
            except Exception as e:
                print(f"Error loading record: {e}")
    
    return records


def get_feedback_text(record: FeedbackRecord) -> str:
    """Convert a feedback record to text for embedding"""
    outcome_rating = record.outcome_rating if record.outcome_rating is not None else \
                    (record.actual_sold / record.recommended_qty) if record.recommended_qty > 0 else 0.5
    
    config_performance = "good" if outcome_rating > 0.8 else "average" if outcome_rating > 0.5 else "poor"
    profitability = "high profit" if record.gross_profit > 2000 else "medium profit" if record.gross_profit > 1000 else "low profit"
    turnover = "fast turn" if record.ddt < 20 else "average turn" if record.ddt < 40 else "slow turn"
    
    forecast_text = ""
    if record.forecast_accuracy is not None:
        forecast_text = f" Forecast accuracy was {record.forecast_accuracy:.2f}, which is {('excellent' if record.forecast_accuracy > 0.9 else 'good' if record.forecast_accuracy > 0.7 else 'fair' if record.forecast_accuracy > 0.5 else 'poor')}."
    
    constraint_text = ""
    if record.constraint_hit:
        constraint_text = " A constraint was hit during this period, which limited sales."
    
    return f"Configuration {record.config_id} sold {record.actual_sold} units with a recommendation of {record.recommended_qty} units on {record.sale_date}. " \
           f"This configuration had {profitability} at ${record.gross_profit:.2f} and {turnover} at {record.ddt} days to turn. " \
           f"Overall performance was {config_performance}.{forecast_text}{constraint_text}"


async def build_or_update_vector_index(records: Optional[List[FeedbackRecord]] = None) -> Dict[str, Any]:
    """Build or update the FAISS vector index for feedback records"""
    # Load existing index if available
    if VECTOR_INDEX_PATH.exists() and VECTOR_METADATA_PATH.exists():
        try:
            index = faiss.read_index(str(VECTOR_INDEX_PATH))
            with open(VECTOR_METADATA_PATH, "r") as f:
                metadata = json.load(f)
            
            # If no new records to add, return the existing metadata
            if records is None or len(records) == 0:
                return {
                    "status": "success", 
                    "message": "Using existing vector index",
                    "dimensions": metadata["dimensions"],
                    "total_records": metadata["total_records"]
                }
                
            # Get the ids of existing records to avoid duplicates
            existing_ids = set(metadata["record_ids"])
            
            # Filter out records that are already in the index
            new_records = [r for r in records if r.record_id not in existing_ids]
            
            if len(new_records) == 0:
                return {
                    "status": "success", 
                    "message": "No new records to add to vector index",
                    "dimensions": metadata["dimensions"],
                    "total_records": metadata["total_records"]
                }
                
            # Add new records to the index
            embeddings = []
            new_metadata = {
                "record_ids": metadata["record_ids"].copy(),
                "config_ids": metadata["config_ids"].copy(),
                "sale_dates": metadata["sale_dates"].copy(),
                "dimensions": metadata["dimensions"],
                "total_records": metadata["total_records"] + len(new_records),
                "last_updated": datetime.now().isoformat()
            }
            
            for record in new_records:
                if record.embedding is None:
                    text = get_feedback_text(record)
                    record.embedding = await create_embedding(text)
                
                embeddings.append(record.embedding)
                new_metadata["record_ids"].append(record.record_id)
                new_metadata["config_ids"].append(record.config_id)
                new_metadata["sale_dates"].append(record.sale_date)
            
            # Convert to numpy array and add to index
            embeddings_array = np.array(embeddings).astype('float32')
            index.add(embeddings_array)
            
            # Save updated index and metadata
            faiss.write_index(index, str(VECTOR_INDEX_PATH))
            with open(VECTOR_METADATA_PATH, "w") as f:
                json.dump(new_metadata, f)
            
            return {
                "status": "success",
                "message": f"Added {len(new_records)} new records to vector index",
                "dimensions": new_metadata["dimensions"],
                "total_records": new_metadata["total_records"]
            }
            
        except Exception as e:
            print(f"Error loading existing index: {e}")
            # If there's an error, rebuild the index from scratch
            pass
    
    # No existing index or error loading it, build from scratch
    if records is None:
        records = load_all_feedback_records()
    
    if len(records) == 0:
        return {
            "status": "error",
            "message": "No records available to build vector index",
            "dimensions": 0,
            "total_records": 0
        }
    
    # Create embeddings for all records
    embeddings = []
    record_ids = []
    config_ids = []
    sale_dates = []
    dimensions = 0
    
    for record in records:
        if record.embedding is None:
            text = get_feedback_text(record)
            record.embedding = await create_embedding(text)
            save_feedback_record(record)  # Update the record with embedding
        
        embeddings.append(record.embedding)
        record_ids.append(record.record_id)
        config_ids.append(record.config_id)
        sale_dates.append(record.sale_date)
        
        if dimensions == 0:
            dimensions = len(record.embedding)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings).astype('float32')
    
    # Create and save the FAISS index
    index = faiss.IndexFlatL2(dimensions)
    index.add(embeddings_array)
    faiss.write_index(index, str(VECTOR_INDEX_PATH))
    
    # Save metadata mapping the index to record IDs
    metadata: VectorMetadata = {
        "record_ids": record_ids,
        "config_ids": config_ids,
        "sale_dates": sale_dates,
        "dimensions": dimensions,
        "total_records": len(record_ids),
        "last_updated": datetime.now().isoformat()
    }
    
    with open(VECTOR_METADATA_PATH, "w") as f:
        json.dump(metadata, f)
    
    return {
        "status": "success",
        "message": f"Built vector index with {len(record_ids)} records",
        "dimensions": dimensions,
        "total_records": len(record_ids)
    }


async def search_feedback(query: FeedbackQuery) -> List[FeedbackSearchResult]:
    """Search the vector index for similar feedback records"""
    if not VECTOR_INDEX_PATH.exists() or not VECTOR_METADATA_PATH.exists():
        return []
    
    # Create embedding for the query
    query_embedding = await create_embedding(query.query_text)
    query_embedding_array = np.array([query_embedding]).astype('float32')
    
    # Load the index and metadata
    index = faiss.read_index(str(VECTOR_INDEX_PATH))
    with open(VECTOR_METADATA_PATH, "r") as f:
        metadata = json.load(f)
    
    # Search the index
    k = min(query.limit, metadata["total_records"])
    if k == 0:
        return []
    
    distances, indices = index.search(query_embedding_array, k)
    
    # Convert results to FeedbackSearchResult
    results = []
    all_records = load_all_feedback_records()
    record_map = {r.record_id: r for r in all_records}
    
    for i, idx in enumerate(indices[0]):
        if idx >= len(metadata["record_ids"]):
            continue
        
        record_id = metadata["record_ids"][idx]
        config_id = metadata["config_ids"][idx]
        sale_date = metadata["sale_dates"][idx]
        
        # Calculate similarity score (1 - normalized distance)
        # L2 distance can be unbounded, so we use a simple normalization
        max_distance = 10.0  # Arbitrary max distance for normalization
        normalized_dist = min(distances[0][i], max_distance) / max_distance
        score = 1.0 - normalized_dist
        
        if score < query.min_score:
            continue
        
        # Create feedback text
        text = ""
        if record_id in record_map:
            text = get_feedback_text(record_map[record_id])
        else:
            text = f"Configuration {config_id} record from {sale_date}"
        
        results.append({
            "record_id": record_id,
            "config_id": config_id, 
            "sale_date": sale_date,
            "text": text,
            "score": score
        })
    
    return results


@function_tool
async def process_feedback_file(ctx: RunContextWrapper[Any], file_path: str) -> Dict[str, Any]:
    """
    Process a feedback CSV file, validate the data, and store it in the feedback store.
    
    Args:
        file_path: Path to the CSV file containing feedback data
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Validate that required columns exist
        required_columns = ['config_id', 'sale_date', 'gross_profit', 'ddt', 'recommended_qty', 'actual_sold']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return {
                "status": "error",
                "message": f"Missing required columns: {', '.join(missing_columns)}"
            }
        
        # Process each row and create feedback records
        records = []
        failed_records = 0
        
        for _, row in df.iterrows():
            try:
                # Create feedback record
                record_data = {
                    "config_id": str(row['config_id']),
                    "sale_date": str(row['sale_date']),
                    "gross_profit": float(row['gross_profit']),
                    "ddt": int(row['ddt']),
                    "recommended_qty": int(row['recommended_qty']),
                    "actual_sold": int(row['actual_sold']),
                }
                
                # Add optional fields if they exist
                if 'constraint_hit' in df.columns:
                    record_data["constraint_hit"] = bool(row['constraint_hit'])
                if 'forecast_accuracy' in df.columns:
                    record_data["forecast_accuracy"] = float(row['forecast_accuracy'])
                if 'outcome_rating' in df.columns:
                    record_data["outcome_rating"] = float(row['outcome_rating'])
                
                # Create and validate the record
                record = FeedbackRecord(**record_data)
                
                # Generate embedding for the record
                text = get_feedback_text(record)
                record.embedding = await create_embedding(text)
                
                # Save to JSONL store
                save_feedback_record(record)
                records.append(record)
                
            except Exception as e:
                print(f"Error processing row: {e}")
                failed_records += 1
        
        # Build or update the vector index
        index_result = await build_or_update_vector_index(records)
        
        # Calculate summary statistics
        if records:
            avg_gross_profit = sum(r.gross_profit for r in records) / len(records)
            avg_ddt = sum(r.ddt for r in records) / len(records)
            
            # Calculate average outcome rating if available
            ratings = [r.outcome_rating for r in records if r.outcome_rating is not None]
            avg_outcome_rating = sum(ratings) / len(ratings) if ratings else None
            
            summary = FeedbackSummary(
                total_records=len(records) + failed_records,
                successful_records=len(records),
                failed_records=failed_records,
                avg_gross_profit=avg_gross_profit,
                avg_ddt=avg_ddt,
                avg_outcome_rating=avg_outcome_rating,
                vectors_created=len(records),
                status="success",
                message=f"Processed {len(records)} records successfully"
            )
            
            return {
                "status": "success",
                "message": f"Processed {len(records)} records with {failed_records} failures",
                "records_processed": len(records),
                "vectors_created": len(records),
                "summary": summary.model_dump(),
                "index_status": index_result["status"]
            }
        else:
            return {
                "status": "error",
                "message": f"No valid records found in the file. Failed to process {failed_records} records."
            }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error processing feedback file: {str(e)}"
        }


@function_tool
async def search_feedback_patterns(ctx: RunContextWrapper[Any], query_text: str, limit: int = 5, min_score: float = 0.7) -> Dict[str, Any]:
    """
    Search the feedback data for patterns matching the query text.
    
    Args:
        query_text: The natural language query to search for
        limit: Maximum number of results to return
        min_score: Minimum similarity score threshold (0-1)
    """
    query = FeedbackQuery(
        query_text=query_text,
        limit=limit,
        min_score=min_score
    )
    
    try:
        results = await search_feedback(query)
        
        if not results:
            return {
                "status": "warning",
                "message": "No matching patterns found",
                "results": []
            }
        
        return {
            "status": "success",
            "message": f"Found {len(results)} matching patterns",
            "results": results
        }
    
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error searching feedback: {str(e)}",
            "results": []
        }


@function_tool
async def get_feedback_statistics(ctx: RunContextWrapper[Any], config_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get statistics about the feedback data.
    
    Args:
        config_id: Optional configuration ID to filter statistics
    """
    records = load_all_feedback_records()
    
    if not records:
        return {
            "status": "warning",
            "message": "No feedback records found",
            "total_records": 0
        }
    
    # Filter by config_id if provided
    if config_id:
        records = [r for r in records if r.config_id == config_id]
        
        if not records:
            return {
                "status": "warning",
                "message": f"No feedback records found for config_id {config_id}",
                "total_records": 0
            }
    
    # Calculate statistics
    avg_gross_profit = sum(r.gross_profit for r in records) / len(records)
    avg_ddt = sum(r.ddt for r in records) / len(records)
    
    # Calculate outcome performance
    performance_ratio = []
    for r in records:
        if r.recommended_qty > 0:
            ratio = r.actual_sold / r.recommended_qty
            performance_ratio.append(ratio)
    
    avg_performance = sum(performance_ratio) / len(performance_ratio) if performance_ratio else None
    
    # Count records with vector embeddings
    vectors_count = sum(1 for r in records if r.embedding is not None)
    
    # Get unique configurations
    unique_configs = set(r.config_id for r in records)
    
    return {
        "status": "success",
        "total_records": len(records),
        "unique_configurations": len(unique_configs),
        "vectors_count": vectors_count,
        "avg_gross_profit": avg_gross_profit,
        "avg_ddt": avg_ddt,
        "avg_performance_ratio": avg_performance,
        "configurations": list(unique_configs)
    }


# Define the feedback collector agent
feedback_collector = Agent(
    name="Feedback Collector",
    description="Processes sales outcome data and builds vector search index",
    instructions="""
    You are an agent that processes sales outcome feedback data for the DOSO AI system.
    Your responsibilities include:
    
    1. Processing CSV files containing sales outcomes
    2. Validating and cleaning the data
    3. Storing records in a persistent JSONL store
    4. Building and maintaining a vector index for pattern discovery
    5. Providing statistics and insights about the feedback data
    
    When processing files, ensure all required fields are present and validate data types.
    When searching for patterns, use the vector search to find similar outcomes.
    """,
    tools=[
        process_feedback_file,
        search_feedback_patterns,
        get_feedback_statistics
    ],
    model="gpt-4o",
)
