#!/usr/bin/env python
"""
DOSO AI Self-Learning System Sample Data Loader

This script loads sample data from CSV files into the PostgreSQL database.
It populates the configurations, feedback, and other tables with realistic
sample data for testing and demonstration purposes.
"""

import os
import sys
import uuid
import json
import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

# Get database connection string
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("Error: DATABASE_URL environment variable not set")
    print("Please set it in the .env file")
    sys.exit(1)

# Define paths
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DATA_DIR = SCRIPT_DIR / "sample_data"

# Check sample data directory exists
if not SAMPLE_DATA_DIR.exists():
    print(f"Creating sample data directory: {SAMPLE_DATA_DIR}")
    os.makedirs(SAMPLE_DATA_DIR, exist_ok=True)


def ensure_sample_data():
    """Ensure sample data files exist, create if not"""
    # Create sample feedback data
    feedback_file = SAMPLE_DATA_DIR / "feedback_sample.csv"
    if not feedback_file.exists():
        print(f"Creating sample feedback data: {feedback_file}")
        # Generate realistic feedback data
        configs = ["config1", "config2", "config3", "config4", "config5"]
        data = []
        
        for config_id in configs:
            # Generate data for each config
            for i in range(10):  # 10 records per config
                sale_date = (datetime.now() - timedelta(days=i*7)).strftime("%Y-%m-%d")
                gross_profit = int(1200 + 800 * (0.5 + 0.5 * (hash(config_id) % 100) / 100))
                ddt = int(25 + 20 * (0.5 + 0.5 * (hash(config_id + str(i)) % 100) / 100))
                recommended_qty = int(8 + 7 * (0.5 + 0.5 * (hash(config_id + "rec" + str(i)) % 100) / 100))
                actual_sold = int(recommended_qty * (0.7 + 0.6 * (hash(config_id + "sold" + str(i)) % 100) / 100))
                outcome_rating = round((0.7 + 0.3 * (hash(config_id + "rating" + str(i)) % 100) / 100), 2)
                
                data.append({
                    "config_id": config_id,
                    "sale_date": sale_date,
                    "gross_profit": gross_profit,
                    "ddt": ddt,
                    "recommended_qty": recommended_qty,
                    "actual_sold": actual_sold,
                    "outcome_rating": outcome_rating
                })
        
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(feedback_file, index=False)
    
    # Create sample sales history data
    sales_file = SAMPLE_DATA_DIR / "sales_sample.csv"
    if not sales_file.exists():
        print(f"Creating sample sales data: {sales_file}")
        # Generate realistic sales data
        configs = ["config1", "config2", "config3", "config4", "config5"]
        data = []
        
        for config_id in configs:
            # Generate weekly sales for the past year
            for i in range(52):  # 52 weeks
                date = (datetime.now() - timedelta(weeks=i)).strftime("%Y-%m-%d")
                # Create seasonal pattern with some randomness
                base_qty = 10 if config_id == "config1" else 8 if config_id == "config2" else 5 if config_id == "config3" else 12 if config_id == "config4" else 4
                seasonal = 3 * ((i % 52) / 13) * (1 if (i % 52) < 26 else -1)  # Seasonal component
                trend = i * 0.05  # Small upward trend
                noise = (hash(config_id + date) % 100) / 100 * 4 - 2  # Random noise between -2 and 2
                
                quantity = max(1, int(base_qty + seasonal + trend + noise))
                
                data.append({
                    "date": date,
                    "config_id": config_id,
                    "quantity": quantity
                })
        
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(sales_file, index=False)
    
    # Create sample market data
    market_file = SAMPLE_DATA_DIR / "market_sample.csv"
    if not market_file.exists():
        print(f"Creating sample market data: {market_file}")
        # Generate realistic market data
        configs = ["config1", "config2", "config3", "config4", "config5"]
        data = []
        
        for config_id in configs:
            # Generate monthly market data for the past year
            for i in range(12):  # 12 months
                date = (datetime.now() - timedelta(days=i*30)).strftime("%Y-%m-%d")
                
                market_share = round(0.05 + 0.15 * (hash(config_id) % 100) / 100, 3)
                competitor_price = int(30000 + 20000 * (hash(config_id + "price") % 100) / 100)
                market_growth = round(0.01 + 0.04 * (hash(config_id + date) % 100) / 100, 3)
                
                data.append({
                    "date": date,
                    "config_id": config_id,
                    "market_share": market_share,
                    "competitor_price": competitor_price,
                    "market_growth": market_growth
                })
        
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(market_file, index=False)
    
    # Create sample inventory data
    inventory_file = SAMPLE_DATA_DIR / "inventory_sample.csv"
    if not inventory_file.exists():
        print(f"Creating sample inventory data: {inventory_file}")
        # Generate realistic inventory data
        configs = ["config1", "config2", "config3", "config4", "config5"]
        data = []
        
        for config_id in configs:
            date = datetime.now().strftime("%Y-%m-%d")
            
            on_hand = int(5 + 20 * (hash(config_id + "onhand") % 100) / 100)
            in_transit = int(3 + 10 * (hash(config_id + "transit") % 100) / 100)
            allocated = int(2 + 8 * (hash(config_id + "alloc") % 100) / 100)
            
            data.append({
                "date": date,
                "config_id": config_id,
                "on_hand": on_hand,
                "in_transit": in_transit,
                "allocated": allocated,
                "available": on_hand - allocated
            })
        
        # Save to CSV
        df = pd.DataFrame(data)
        df.to_csv(inventory_file, index=False)
    
    return feedback_file, sales_file, market_file, inventory_file


def load_data_to_database(feedback_file, sales_file, market_file, inventory_file):
    """Load sample data into the database"""
    print(f"Connecting to database: {DATABASE_URL}")
    
    try:
        # Create database engine
        engine = create_engine(DATABASE_URL)
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            print("Database connection successful")
        
        # Load feedback data
        print(f"Loading feedback data from {feedback_file}")
        feedback_df = pd.read_csv(feedback_file)
        
        # Insert feedback data
        with engine.connect() as conn:
            for _, row in feedback_df.iterrows():
                # Check if the record already exists
                result = conn.execute(
                    text("SELECT id FROM feedback WHERE config_id = :config_id AND sale_date = :sale_date"),
                    {"config_id": row['config_id'], "sale_date": row['sale_date']}
                )
                existing = result.fetchone()
                
                if not existing:
                    # Insert new record
                    conn.execute(
                        text("""
                            INSERT INTO feedback 
                            (id, config_id, sale_date, gross_profit, ddt, recommended_qty, actual_sold, outcome_rating)
                            VALUES (:id, :config_id, :sale_date, :gross_profit, :ddt, :recommended_qty, :actual_sold, :outcome_rating)
                        """),
                        {
                            "id": str(uuid.uuid4()),
                            "config_id": row['config_id'],
                            "sale_date": row['sale_date'],
                            "gross_profit": row['gross_profit'],
                            "ddt": row['ddt'],
                            "recommended_qty": row['recommended_qty'],
                            "actual_sold": row['actual_sold'],
                            "outcome_rating": row['outcome_rating']
                        }
                    )
            
            conn.commit()
        
        print("Sample data loaded successfully")
        return True
    
    except Exception as e:
        print(f"Error loading data to database: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="DOSO AI Sample Data Loader")
    parser.add_argument("--force", action="store_true", help="Force recreate sample data files")
    parser.add_argument("--verify", action="store_true", help="Verify database connection only")
    args = parser.parse_args()
    
    print("DOSO AI Self-Learning System Sample Data Loader")
    print("===============================================")
    
    if args.verify:
        try:
            engine = create_engine(DATABASE_URL)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                print("✓ Database connection verified successfully")
            return 0
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            return 1
    
    # Ensure sample data exists
    if args.force:
        print("Forcing recreation of sample data files")
        for file in SAMPLE_DATA_DIR.glob("*_sample.csv"):
            os.remove(file)
    
    feedback_file, sales_file, market_file, inventory_file = ensure_sample_data()
    
    # Load data to database
    success = load_data_to_database(feedback_file, sales_file, market_file, inventory_file)
    
    if success:
        print("✓ Sample data loaded successfully")
        return 0
    else:
        print("✗ Failed to load sample data")
        return 1


if __name__ == "__main__":
    sys.exit(main())
