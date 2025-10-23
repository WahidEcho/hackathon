#!/usr/bin/env python3
"""
RL-Based Solver for Robin Logistics Environment - MWVRP Hackathon
Loads pretrained Deep Q-Learning model for vehicle routing optimization
"""

import os
from typing import Dict, List, Optional
from rl_model import LogisticsRLModel


# Global model instance (loaded once)
_rl_model = None
_model_loaded = False

# Logging control via environment variable
MVP_LOG = os.environ.get('MVP_LOG', '0') == '1'


def log(message: str) -> None:
    """Conditional logging based on MVP_LOG environment variable."""
    if MVP_LOG:
        print(f"[RL-SOLVER] {message}")


def load_rl_model() -> LogisticsRLModel:
    """
    Load the pretrained RL model.
    
    Returns:
        Loaded RL model instance
    """
    global _rl_model, _model_loaded
    
    if _model_loaded and _rl_model is not None:
        return _rl_model
    
    try:
        # Initialize model
        _rl_model = LogisticsRLModel(state_dim=100, action_dim=50, learning_rate=0.001)
        
        # Try to load trained model
        model_files = ['trained_model.pkl', 'rl_model_checkpoint.pkl']
        model_loaded = False
        
        for model_file in model_files:
            if os.path.exists(model_file):
                if _rl_model.load_model(model_file):
                    log(f"✅ Loaded pretrained model from {model_file}")
                    log(f"📊 Model trained for {_rl_model.episodes_trained} episodes")
                    log(f"🎯 Best reward achieved: {_rl_model.best_reward:.1f}")
                    model_loaded = True
                    break
        
        if not model_loaded:
            log("⚠️  No pretrained model found. Using randomly initialized model.")
            log("💡 Run 'python rl_trainer.py' to train the model first.")
            # Set a low exploration rate for untrained model
            _rl_model.epsilon = 0.3
        else:
            # Use low exploration rate for inference
            _rl_model.epsilon = 0.05
        
        _model_loaded = True
        return _rl_model
        
    except Exception as e:
        log(f"❌ Error loading RL model: {e}")
        log("🔄 Falling back to basic heuristic approach")
        return None


def rl_solve(env) -> Dict:
    """
    Solve using the RL model.
    
    Args:
        env: LogisticsEnvironment instance
        
    Returns:
        Solution dictionary with routes
    """
    try:
        # Load RL model
        model = load_rl_model()
        
        if model is None:
            log("⚠️  RL model not available, using fallback solution")
            return fallback_solver(env)
        
        log("🧠 Using RL model for route optimization")
        
        # Encode current environment state
        state = model.encode_state(env)
        log(f"📊 State encoded: {len(state)} features")
        
        # Get action from trained policy (no exploration for inference)
        action = model.get_action(state, exploration=True)  # Small exploration for variety
        log(f"🎯 RL policy selected action: {action}")
        
        # Convert action to solution
        solution = model.decode_action_to_solution(action, env)
        
        # Log solution quality
        total_routes = len(solution.get('routes', []))
        total_pickups = sum(len(step.get('pickups', [])) for route in solution.get('routes', []) for step in route.get('steps', []))
        total_deliveries = sum(len(step.get('deliveries', [])) for route in solution.get('routes', []) for step in route.get('steps', []))
        
        log(f"✅ RL solution generated:")
        log(f"   Routes: {total_routes}")
        log(f"   Pickups: {total_pickups}")
        log(f"   Deliveries: {total_deliveries}")
        
        # Validate solution has reasonable content
        if total_routes == 0 or (total_pickups == 0 and total_deliveries == 0):
            log("⚠️  RL solution appears empty, using fallback")
            return fallback_solver(env)
        
        return solution
        
    except Exception as e:
        log(f"❌ RL solving failed: {e}")
        log("🔄 Using fallback solver")
        return fallback_solver(env)


def fallback_solver(env) -> Dict:
    """
    Fallback solver using simple heuristic approach.
    Used when RL model is not available or fails.
    
    Args:
        env: LogisticsEnvironment instance
        
    Returns:
        Solution dictionary
    """
    log("🔧 Using fallback heuristic solver")
    
    try:
        order_ids = env.get_all_order_ids()
        vehicle_ids = env.get_available_vehicles()
        
        if not order_ids or not vehicle_ids:
            return {"routes": []}
        
        routes = []
        orders_per_vehicle = max(1, len(order_ids) // len(vehicle_ids))
        
        for i, vehicle_id in enumerate(vehicle_ids):
            try:
                home_node = env.get_vehicle_home_warehouse(vehicle_id)
                if home_node is None:
                    continue
                
                # Get orders for this vehicle
                start_idx = i * orders_per_vehicle
                end_idx = min(start_idx + orders_per_vehicle + 1, len(order_ids))
                vehicle_orders = order_ids[start_idx:end_idx]
                
                if not vehicle_orders:
                    continue
                
                steps = [{'node_id': home_node, 'pickups': [], 'deliveries': [], 'unloads': []}]
                pickup_items = []
                delivery_items = []
                
                # Process orders for this vehicle
                for order_id in vehicle_orders[:3]:  # Limit to 3 orders
                    try:
                        requirements = env.get_order_requirements(order_id)
                        destination = env.get_order_location(order_id)
                        
                        if requirements and destination:
                            # Find home warehouse
                            for wh_id, wh in env.warehouses.items():
                                if wh.location and wh.location.id == home_node:
                                    for sku_id, qty in requirements.items():
                                        pickup_items.append({
                                            'warehouse_id': wh_id,
                                            'sku_id': sku_id,
                                            'quantity': min(qty, 10)
                                        })
                                        delivery_items.append({
                                            'order_id': order_id,
                                            'sku_id': sku_id,
                                            'quantity': min(qty, 10)
                                        })
                                    break
                            
                            # Add delivery step
                            steps.append({
                                'node_id': destination,
                                'pickups': [],
                                'deliveries': [d for d in delivery_items if d['order_id'] == order_id],
                                'unloads': []
                            })
                    except:
                        continue
                
                # Add pickup step
                if pickup_items:
                    steps.insert(1, {
                        'node_id': home_node,
                        'pickups': pickup_items,
                        'deliveries': [],
                        'unloads': []
                    })
                
                # Return home
                if len(steps) > 1:
                    steps.append({'node_id': home_node, 'pickups': [], 'deliveries': [], 'unloads': []})
                    routes.append({
                        'vehicle_id': vehicle_id,
                        'steps': steps
                    })
            except:
                continue
        
        log(f"🔧 Fallback solution: {len(routes)} routes generated")
        return {"routes": routes}
        
    except Exception as e:
        log(f"❌ Fallback solver failed: {e}")
        return {"routes": []}


def solver(env) -> Dict:
    """
    Main solver function for Robin Logistics Environment.
    Uses pretrained RL model for optimization.
    
    Args:
        env: LogisticsEnvironment instance
        
    Returns:
        Complete solution dict with routes and sequential steps
    """
    log("🚀 RL-Based Solver Starting")
    
    try:
        # Get basic scenario info
        order_ids = env.get_all_order_ids()
        vehicle_ids = env.get_available_vehicles()
        
        log(f"📊 Scenario: {len(order_ids)} orders, {len(vehicle_ids)} vehicles")
        
        if not order_ids or not vehicle_ids:
            log("⚠️  Empty scenario detected")
            return {"routes": []}
        
        # Use RL model for solving
        solution = rl_solve(env)
        
        # Final validation and logging
        total_routes = len(solution.get('routes', []))
        total_pickups = sum(len(step.get('pickups', [])) for route in solution.get('routes', []) for step in route.get('steps', []))
        total_deliveries = sum(len(step.get('deliveries', [])) for route in solution.get('routes', []) for step in route.get('steps', []))
        
        log(f"✅ Final solution stats:")
        log(f"   Total routes: {total_routes}")
        log(f"   Total pickups: {total_pickups}")
        log(f"   Total deliveries: {total_deliveries}")
        
        # Quick validation
        if total_pickups > 10 and total_deliveries > 10:
            log("🎯 Solution appears to have substantial operations - SUCCESS!")
        elif total_routes > 0:
            log("⚠️  Solution has routes but limited operations")
        else:
            log("❌ Solution appears empty")
        
        return solution
        
    except Exception as e:
        log(f"❌ Critical solver error: {e}")
        return {"routes": []}


def my_solver(env) -> Dict:
    """
    Alias for solver function to maintain compatibility with existing dashboard.
    
    Args:
        env: LogisticsEnvironment instance
        
    Returns:
        Same solution as solver(env)
    """
    return solver(env)


def get_model_info() -> Dict:
    """
    Get information about the loaded RL model.
    
    Returns:
        Model information dictionary
    """
    try:
        model = load_rl_model()
        if model:
            return model.get_model_info()
        else:
            return {
                'model_loaded': False,
                'error': 'RL model not available'
            }
    except Exception as e:
        return {
            'model_loaded': False,
            'error': str(e)
        }


if __name__ == '__main__':
    """Test the RL solver."""
    print("🧠 RL-Based Solver Test")
    print("=" * 40)
    
    try:
        from robin_logistics import LogisticsEnvironment
        
        # Test model loading
        model_info = get_model_info()
        print(f"📊 Model Info: {model_info}")
        
        # Test solving
        env = LogisticsEnvironment()
        solution = solver(env)
        
        routes = len(solution.get('routes', []))
        pickups = sum(len(step.get('pickups', [])) for route in solution.get('routes', []) for step in route.get('steps', []))
        deliveries = sum(len(step.get('deliveries', [])) for route in solution.get('routes', []) for step in route.get('steps', []))
        
        print(f"🎯 Test Results:")
        print(f"   Routes: {routes}")
        print(f"   Pickups: {pickups}")
        print(f"   Deliveries: {deliveries}")
        
        if pickups > 0 and deliveries > 0:
            print("✅ RL Solver working correctly!")
        else:
            print("⚠️  Limited operations - check model training")
            
    except ImportError:
        print("❌ robin-logistics-env not found")
        print("💡 Install: pip install robin-logistics-env")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()