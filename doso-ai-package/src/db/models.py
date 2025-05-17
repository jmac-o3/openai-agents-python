"""
SQLAlchemy models for DOSO AI
"""

from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class MarketAnalysisRecord(Base):
    """Market analysis database record"""

    __tablename__ = "market_analyses"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    data_freshness_score = Column(Float, nullable=False)
    confidence_score = Column(Float, nullable=False)
    analyzed_segments = Column(JSON, nullable=False)
    segment_trends = Column(JSON, nullable=False)
    demand_forecast = Column(JSON, nullable=False)
    allocation_recommendations = Column(JSON, nullable=False)
    risk_factors = Column(JSON, nullable=False)
    opportunities = Column(JSON, nullable=False)


class FastTurnRecord(Base):
    """Fast Turn report data record"""

    __tablename__ = "fast_turn_data"

    id = Column(Integer, primary_key=True)
    model_code = Column(String, nullable=False)
    model_year = Column(Integer, nullable=False)
    region = Column(String, nullable=False)
    days_supply = Column(Float, nullable=False)
    turn_rate = Column(Float, nullable=False)
    market_demand_score = Column(Float, nullable=False)
    allocation_units = Column(Integer, nullable=False)
    dealer_orders = Column(Integer)
    timestamp = Column(DateTime, nullable=False)

    analysis_id = Column(Integer, ForeignKey("market_analyses.id"))
    analysis = relationship("MarketAnalysisRecord", back_populates="fast_turn_data")


# Add the back-reference
MarketAnalysisRecord.fast_turn_data = relationship(
    "FastTurnRecord",
    back_populates="analysis",
    order_by=FastTurnRecord.timestamp.desc(),
)
