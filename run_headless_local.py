#!/usr/bin/env python3
"""
Headless Local Runner for Robin Logistics MVP
Executes the solver locally and reports results without dashboard.
"""

import sys
import traceback
from typing import Dict, Any


def print_solution_summary(solution: Dict[str, Any], env) -> None:
    """Print comprehensive solution summary."""
    routes = solution.get('routes', [])
    
    print("\n" + "="*60)
    print("SOLUTION SUMMARY")
    print("="*60)
    
    print(f"Total Routes: {len(routes)}")
    
    # Route details
    active_routes = 0
    total_steps = 0
    total_pickups = 0
    total_deliveries = 0
    
    for route in routes:
        steps = route.get('steps', [])
        if len(steps) > 1:  # More than just home node
            active_routes += 1
        
        total_steps += len(steps)
        
        for step in steps:
            total_pickups += len(step.get('pickups', []))
            total_deliveries += len(step.get('deliveries', []))
    
    print(f"Active Routes: {active_routes}")
    print(f"Total Steps: {total_steps}")
    print(f"Total Pickup Operations: {total_pickups}")
    print(f"Total Delivery Operations: {total_deliveries}")
    
    # Cost calculation
    try:
        cost = env.calculate_solution_cost(solution)
        print(f"Total Cost: ${cost:.2f}")
    except Exception as e:
        print(f"Cost calculation failed: {e}")
    
    # Fulfillment analysis
    try:
        fulfillment = env.get_solution_fulfillment_summary(solution)
        if fulfillment:
            total_requested = fulfillment.get('total_requested', 0)
            total_delivered = fulfillment.get('total_delivered', 0)
            fulfillment_rate = (total_delivered / total_requested * 100) if total_requested > 0 else 0
            
            print(f"\nFULFILLMENT ANALYSIS:")
            print(f"Total Items Requested: {total_requested}")
            print(f"Total Items Delivered: {total_delivered}")
            print(f"Fulfillment Rate: {fulfillment_rate:.1f}%")
            print(f"Unfulfilled Items: {total_requested - total_delivered}")
        else:
            print("\nFulfillment data not available")
            
    except Exception as e:
        print(f"Fulfillment analysis failed: {e}")
    
    print("="*60)


def main() -> int:
    """Main execution function."""
    print("Robin Logistics MVP - Headless Local Runner")
    print("=" * 50)
    
    try:
        # Initialize environment
        print("Initializing environment...")
        from robin_logistics import LogisticsEnvironment
        env = LogisticsEnvironment()
        print("✅ Environment initialized successfully")
        
        # Import and run solver
        print("Importing solver...")
        import solver
        print("✅ Solver imported successfully")
        
        print("Executing solver...")
        solution = solver.solver(env)
        print("✅ Solver execution completed")
        
        # Validate solution
        print("Validating solution...")
        is_valid, message = env.validate_solution_complete(solution)
        
        if is_valid:
            print("✅ Solution is VALID")
        else:
            print(f"❌ Solution is INVALID: {message}")
        
        # Print results
        print_solution_summary(solution, env)
        
        # Return appropriate exit code
        return 0 if is_valid else 1
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure 'robin-logistics-env' is installed: pip install robin-logistics-env")
        return 2
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    exit_code = main()
    print(f"\nExiting with code: {exit_code}")
    sys.exit(exit_code)
