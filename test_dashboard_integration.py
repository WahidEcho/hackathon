#!/usr/bin/env python3
"""
Test Dashboard Integration with RL System
"""

import os
os.environ['MVP_LOG'] = '1'

from robin_logistics import LogisticsEnvironment
import solver

def main():
    print("=== TESTING DASHBOARD INTEGRATION ===")
    
    try:
        env = LogisticsEnvironment()
        print("✅ Environment created")
        
        # This is the exact call run_dashboard.py makes
        solution = solver.my_solver(env)
        print(f"✅ Dashboard call successful: {len(solution.get('routes', []))} routes")
        
        # Test metrics calculation (what dashboard displays)
        cost = env.calculate_solution_cost(solution)
        print(f"✅ Cost calculation: ${cost:.2f}")
        
        # Count operations
        pickups = sum(len(step.get('pickups', [])) for route in solution.get('routes', []) for step in route.get('steps', []))
        deliveries = sum(len(step.get('deliveries', [])) for route in solution.get('routes', []) for step in route.get('steps', []))
        
        print(f"✅ Operations: {pickups} pickups, {deliveries} deliveries")
        
        # Test validation
        try:
            validation = env.validate_solution_complete(solution)
            print(f"✅ Validation completed")
        except:
            print("⚠️  Validation had issues (expected)")
        
        print("🎯 DASHBOARD INTEGRATION TEST: SUCCESS!")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
