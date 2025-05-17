---
applyTo: '**'
---
Coding standards, domain knowledge, and preferences that AI should follow.

# DOSO AI Comprehensive Build Document

## Executive Summary
The Dealer Inventory Optimization System (DOSO AI) is an advanced, AI-driven web application for Ford dealerships to optimize new vehicle inventory using OpenAI Agents SDK, multi-agent orchestration, and advanced analytics.

## Project Overview
- Integrates sales history, inventory, order bank, and Ford operational reports.
- Multi-agent architecture: specialized agents for parsing, analysis, recommendations.
- Innovations: RAG with vector stores, function calling, hybrid orchestration, evaluation framework.

## Key Objectives
1. Maximize dealer profitability via optimal configuration mix.
2. Accelerate inventory turn (reduce DDT).
3. Optimize SVA metrics for better allocation.
4. Automate complex analysis (no manual spreadsheets).
5. Streamline ordering workflow (direct WBDO-ready outputs).
6. Enhance decision support with transparent justifications.

## Core Technology Stack
- Backend: Python (FastAPI)
- Database: PostgreSQL + pgvector
- Frontend: React + Tailwind CSS
- AI: OpenAI Agents SDK, OpenAI Vector Stores API
- ML: Prophet, scikit-learn
- Infra: Kubernetes, Prometheus, Grafana, OpenTelemetry

## System Architecture
- Data Layer: PostgreSQL, pgvector, Redis, file storage
- API Layer: FastAPI, REST, auth, rate limiting
- Agent Layer: OpenAI Agents SDK, custom tools, handoffs
- Business Logic: Inventory optimization, SVA, forecasting
- Presentation: React, Tailwind, Redux, Recharts

## Data Flow
1. Input: Upload Ford reports (CSV/PDF), paste emails, request analysis.
2. Processing: Parse, validate, normalize, embed, index.
3. Analysis: SVA, profit/velocity, market, forecasting, inventory, constraints.
4. Recommendation: DOSO scoring, optimization, constraint satisfaction, justification.
5. Output: Dashboard, reports, WBDO-ready lists, alerts, chat guidance.

## Agent Architecture
- Triage Agent: Orchestrates, routes requests.
- Data Processing Agents: CSV, PDF, Email parsing, validation.
- Analysis Agents: SVA, profit/velocity, market, forecasting, inventory, gap, allocation, order bank, constraint check, recommendation.
- Guidance Agent: Explanations, chat, alerts.

## Database Schema
- See detailed SQL tables for configurations, sales_history, inventory_snapshots, order_bank_status, allocation_data, constraints, sva_metrics, configuration_metrics, forecasts, stocking_gaps, recommendations, users, user_preferences, system_logs, file_uploads, chat_history, vector_embeddings.

## Agent Specifications
- Each agent has clear instructions, tools, input/output Pydantic models.
- See detailed Python async function signatures and output models for each agent.

## Data Processing Standards
- CSV: Use pandas, validate columns, normalize models/colors/dates/numbers, report validation issues.
- PDF: Use pdfplumber, PyPDF2, OCR fallback, extract tables/text, validate/normalize.
- Email: LLM-based extraction for allocations, constraints, deadlines; output structured JSON.

## Data Validation & Normalization
- Normalize dates (ISO 8601), status values, numeric fields.
- Cross-file consistency checks (allocation vs scheduling).
- Model/trim mapping for consistency.

## Core Algorithms
- SVA: SVA = (Last Month Sales ÷ Month-End Stock) × 100
- DOSO Score: 0.4*ProfitRank + 0.4*DTTRank_Inverted + 0.2*MarketRank
- Forecasting: Prophet, weekly, with confidence intervals.
- Constraint satisfaction: Adjust recommendations to fit allocation/model/trim limits.

## UI Specifications
- Dashboard: KPIs, widgets for SVA, inventory, order bank, allocation, tasks, alerts.
- SVA Performance: Trend charts, regional comparison, WS2 impact.
- OTD Config Selection: Table, copy/download for WBDO, justification tooltips.

## Coding Standards
- Python: Type hints, Pydantic models, async/await, docstrings, error handling.
- React: Functional components, hooks, prop types, modular widgets.
- SQL: Use constraints, indexes, foreign keys, timestamps.

## Preferences
- All outputs must be ready for direct use in dealership workflows.
- Recommendations must be justified with data and constraints.
- All agent outputs must be structured, validated, and explainable.