-- DOSO AI Self-Learning System Database Initialization
-- This script initializes the PostgreSQL database with the required schema

-- Enable the vector extension for embedding storage and search
CREATE EXTENSION IF NOT EXISTS "vector";

-- Configuration table to store different vehicle configurations
CREATE TABLE IF NOT EXISTS configurations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    parameters JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Forecasts table to store demand forecasts
CREATE TABLE IF NOT EXISTS forecasts (
    id TEXT PRIMARY KEY,
    config_id TEXT REFERENCES configurations(id),
    model_type TEXT NOT NULL,
    data JSONB NOT NULL,
    metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feedback table to store performance outcomes and embeddings
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    config_id TEXT REFERENCES configurations(id),
    sale_date DATE NOT NULL,
    gross_profit NUMERIC NOT NULL,
    ddt INTEGER NOT NULL,
    recommended_qty INTEGER NOT NULL,
    actual_sold INTEGER NOT NULL,
    outcome_rating NUMERIC NOT NULL,
    embedding vector(1536),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on config_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_feedback_config_id ON feedback(config_id);

-- Learning cycles table to track optimization runs
CREATE TABLE IF NOT EXISTS learning_cycles (
    id TEXT PRIMARY KEY,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    optimization_target TEXT NOT NULL,
    forecast_model TEXT NOT NULL,
    learning_model TEXT NOT NULL,
    old_weights JSONB,
    new_weights JSONB,
    improvement NUMERIC,
    status TEXT NOT NULL,
    details JSONB
);

-- Weights table to store the current optimization weights
CREATE TABLE IF NOT EXISTS weights (
    id TEXT PRIMARY KEY,
    profit_weight NUMERIC NOT NULL,
    ddt_weight NUMERIC NOT NULL,
    market_weight NUMERIC NOT NULL,
    forecast_weight NUMERIC NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default weights if not exists
INSERT INTO weights (id, profit_weight, ddt_weight, market_weight, forecast_weight)
VALUES ('default', 0.25, 0.25, 0.25, 0.25)
ON CONFLICT (id) DO NOTHING;

-- Recommendations table to store generated recommendations
CREATE TABLE IF NOT EXISTS recommendations (
    id TEXT PRIMARY KEY,
    config_id TEXT REFERENCES configurations(id),
    recommended_qty INTEGER NOT NULL,
    confidence NUMERIC NOT NULL,
    reasoning TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    weights_id TEXT REFERENCES weights(id)
);

-- Sample data for testing
INSERT INTO configurations (id, name, description, parameters)
VALUES 
    ('config1', 'F-150 XLT', 'Ford F-150 XLT Trim', '{"model": "F-150", "trim": "XLT", "options": ["tow package", "4x4"]}'),
    ('config2', 'F-150 Lariat', 'Ford F-150 Lariat Trim', '{"model": "F-150", "trim": "Lariat", "options": ["luxury package", "4x4"]}'),
    ('config3', 'Explorer XLT', 'Ford Explorer XLT Trim', '{"model": "Explorer", "trim": "XLT", "options": ["third row", "awd"]}'),
    ('config4', 'Bronco Wildtrak', 'Ford Bronco Wildtrak Trim', '{"model": "Bronco", "trim": "Wildtrak", "options": ["sasquatch package", "modular hardtop"]}'),
    ('config5', 'Mustang GT', 'Ford Mustang GT', '{"model": "Mustang", "trim": "GT", "options": ["performance package", "active exhaust"]}')
ON CONFLICT (id) DO NOTHING;
