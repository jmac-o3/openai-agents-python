import asyncio
from datetime import datetime, timedelta
from .order_bank_manager import analyze_dealer_orders

async def main():
    # Example: Analyze last 30 days of orders for a dealer
    dealer_id = "DEMO001"  # This would be your actual dealer ID
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    # Run analysis
    result = await analyze_dealer_orders(
        dealer_id=dealer_id,
        start_date=start_date,
        end_date=end_date,
        include_historical=True
    )
    
    # Print results
    print("\n=== Order Bank Analysis Results ===\n")
    print(f"Total Orders: {result.metrics.total_orders}")
    print(f"Average Turn Rate: {result.metrics.average_turn_rate:.2f}")
    print("\nPopular Configurations:")
    for config in result.metrics.popular_configurations:
        print(f"- {config}")
    
    print("\nRecommendations:")
    for rec in result.recommendations:
        print(f"\nConfiguration: {rec.config_id}")
        print(f"Current Volume: {rec.current_volume}")
        print(f"Recommended Volume: {rec.recommended_volume}")
        print(f"Priority Score: {rec.priority:.2f}")
        print(f"Reasoning: {rec.reasoning}")
    
    print(f"\nSummary:\n{result.summary}")

if __name__ == "__main__":
    asyncio.run(main())
