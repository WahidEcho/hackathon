#!/usr/bin/env python3
"""
MVP Solver for Robin Logistics Environment - MWVRP Hackathon
Inventory-aware greedy algorithm with multi-warehouse pickups and partial fulfillment.
"""

import os
import math
from typing import Dict, List, Tuple, Optional, Any


# Configuration constants
ALPHA_DISTANCE = 0.1        # Distance penalty weight in order scoring
BETA_VOLUME = 2.0          # Volume vs weight ratio in order scoring
MAX_WAREHOUSES_PER_ORDER = 5    # Limit warehouses per order to prevent explosion
MAX_STEPS_PER_VEHICLE = 50      # Safety limit on route complexity
ORDER_PER_VEHICLE_LIMIT = 20    # Max orders per vehicle
CAPACITY_SAFETY_MARGIN = 0.95   # Use 95% of capacity to avoid edge cases
MIN_PICKUP_QUANTITY = 1         # Minimum quantity to consider picking up

# Logging control via environment variable
MVP_LOG = os.environ.get('MVP_LOG', '0') == '1'


def log(message: str) -> None:
    """Conditional logging based on MVP_LOG environment variable."""
    if MVP_LOG:
        print(f"[MVP] {message}")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Haversine distance between two points in kilometers."""
    if lat1 == 0.0 and lon1 == 0.0 and lat2 == 0.0 and lon2 == 0.0:
        return 0.0  # Both points are at origin fallback
    
    R = 6371.0  # Earth radius in km
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def node_latlon(env, node_id: int) -> Tuple[float, float]:
    """Get node coordinates with safe fallbacks."""
    try:
        if node_id in env.nodes:
            node = env.nodes[node_id]
            return (float(node.lat), float(node.lon))
    except (AttributeError, KeyError, ValueError, TypeError):
        pass
    return (0.0, 0.0)  # Safe fallback


def get_sku_specs(env, sku_id: str) -> Dict[str, float]:
    """Get SKU weight and volume with safe defaults."""
    try:
        details = env.get_sku_details(sku_id)
        if details:
            return {
                'weight': float(details.get('weight', 1.0)),
                'volume': float(details.get('volume', 0.1))
            }
    except (AttributeError, KeyError, ValueError, TypeError):
        pass
    return {'weight': 1.0, 'volume': 0.1}  # Safe defaults


def compute_order_score(env, order_id: str, shadow_inventory: Dict[str, Dict[str, int]]) -> float:
    """Compute order priority score (lower = higher priority)."""
    try:
        # Get order details
        order_requirements = env.get_order_requirements(order_id)
        if not order_requirements:
            return float('inf')
        
        order_node = env.get_order_location(order_id)
        if order_node is None:
            return float('inf')
            
        order_lat, order_lon = node_latlon(env, order_node)
        
        # Find nearest warehouse with any required SKU
        min_distance = float('inf')
        total_weight = 0.0
        total_volume = 0.0
        
        for sku_id, quantity in order_requirements.items():
            sku_specs = get_sku_specs(env, sku_id)
            total_weight += sku_specs['weight'] * quantity
            total_volume += sku_specs['volume'] * quantity
            
            # Find warehouses with this SKU
            for wh_id, inventory in shadow_inventory.items():
                if sku_id in inventory and inventory[sku_id] > 0:
                    try:
                        wh = env.get_warehouse_by_id(wh_id)
                        if wh and wh.location:
                            wh_lat, wh_lon = node_latlon(env, wh.location.id)
                            distance = haversine_distance(order_lat, order_lon, wh_lat, wh_lon)
                            min_distance = min(min_distance, distance)
                    except (AttributeError, KeyError, TypeError):
                        continue
        
        if min_distance == float('inf'):
            min_distance = 1000.0  # Large penalty for unfulfillable orders
        
        # Scoring formula: distance + size penalty
        score = min_distance + ALPHA_DISTANCE * (total_weight + BETA_VOLUME * total_volume)
        return score
        
    except Exception:
        return float('inf')


def plan_vehicle_route(env, vehicle_id: str, orders_data: List[Dict], shadow_inventory: Dict[str, Dict[str, int]]) -> List[Dict]:
    """Plan route for a single vehicle using greedy heuristic."""
    try:
        vehicle = env.get_vehicle_by_id(vehicle_id)
        if not vehicle:
            log(f"Vehicle {vehicle_id} not found")
            return []
        
        home_node = env.get_vehicle_home_warehouse(vehicle_id)
        if home_node is None:
            log(f"Home warehouse not found for vehicle {vehicle_id}")
            return []
        
        # Initialize route with home step
        steps = [{'node_id': home_node, 'pickups': [], 'deliveries': [], 'unloads': []}]
        current_node = home_node
        
        # Track remaining capacity
        remaining_weight, remaining_volume = env.get_vehicle_remaining_capacity(vehicle_id)
        remaining_weight *= CAPACITY_SAFETY_MARGIN
        remaining_volume *= CAPACITY_SAFETY_MARGIN
        
        # Track processed orders
        processed_orders = set()
        step_count = 1  # Already have home step
        
        log(f"Vehicle {vehicle_id} starting at node {home_node} with capacity ({remaining_weight:.1f}kg, {remaining_volume:.1f}m³)")
        
        for order_attempt in range(ORDER_PER_VEHICLE_LIMIT):
            if step_count >= MAX_STEPS_PER_VEHICLE - 1:  # Save room for return home
                break
                
            # Find closest unprocessed order
            best_order = None
            best_distance = float('inf')
            current_lat, current_lon = node_latlon(env, current_node)
            
            for order_data in orders_data:
                order_id = order_data['id']
                if order_id in processed_orders:
                    continue
                    
                order_node = order_data['destination']
                order_lat, order_lon = node_latlon(env, order_node)
                distance = haversine_distance(current_lat, current_lon, order_lat, order_lon)
                
                if distance < best_distance:
                    best_distance = distance
                    best_order = order_data
            
            if not best_order:
                break
                
            order_id = best_order['id']
            order_requirements = best_order['requirements']
            order_node = best_order['destination']
            
            # Plan pickups for this order
            pickups_by_warehouse = {}
            total_picked_weight = 0.0
            total_picked_volume = 0.0
            
            for sku_id, needed_qty in order_requirements.items():
                if needed_qty <= 0:
                    continue
                    
                sku_specs = get_sku_specs(env, sku_id)
                
                # Find warehouses with this SKU, prioritize by distance
                warehouse_distances = []
                current_lat, current_lon = node_latlon(env, current_node)
                
                for wh_id, inventory in shadow_inventory.items():
                    if sku_id not in inventory or inventory[sku_id] < MIN_PICKUP_QUANTITY:
                        continue
                        
                    try:
                        wh = env.get_warehouse_by_id(wh_id)
                        if wh and wh.location:
                            wh_lat, wh_lon = node_latlon(env, wh.location.id)
                            distance = haversine_distance(current_lat, current_lon, wh_lat, wh_lon)
                            warehouse_distances.append((distance, wh_id, wh.location.id))
                    except (AttributeError, KeyError, TypeError):
                        continue
                
                # Sort by distance and allocate
                warehouse_distances.sort()
                remaining_needed = needed_qty
                warehouses_used = 0
                
                for distance, wh_id, wh_node in warehouse_distances:
                    if remaining_needed <= 0 or warehouses_used >= MAX_WAREHOUSES_PER_ORDER:
                        break
                        
                    available_qty = shadow_inventory[wh_id][sku_id]
                    pickup_qty = min(remaining_needed, available_qty)
                    
                    # Check capacity constraints
                    pickup_weight = pickup_qty * sku_specs['weight']
                    pickup_volume = pickup_qty * sku_specs['volume']
                    
                    if (total_picked_weight + pickup_weight <= remaining_weight and 
                        total_picked_volume + pickup_volume <= remaining_volume):
                        
                        # Add to pickup plan
                        if wh_node not in pickups_by_warehouse:
                            pickups_by_warehouse[wh_node] = {'warehouse_id': wh_id, 'items': []}
                        
                        pickups_by_warehouse[wh_node]['items'].append({
                            'warehouse_id': wh_id,
                            'sku_id': sku_id,
                            'quantity': pickup_qty
                        })
                        
                        # Update tracking
                        shadow_inventory[wh_id][sku_id] -= pickup_qty
                        total_picked_weight += pickup_weight
                        total_picked_volume += pickup_volume
                        remaining_needed -= pickup_qty
                        remaining_weight -= pickup_weight
                        remaining_volume -= pickup_volume
                        warehouses_used += 1
            
            # If no items could be picked for this order, skip it
            if not pickups_by_warehouse:
                processed_orders.add(order_id)
                continue
            
            # Create pickup steps
            for wh_node, pickup_data in pickups_by_warehouse.items():
                steps.append({
                    'node_id': wh_node,
                    'pickups': pickup_data['items'],
                    'deliveries': [],
                    'unloads': []
                })
                current_node = wh_node
                step_count += 1
                
                if step_count >= MAX_STEPS_PER_VEHICLE - 1:
                    break
            
            # Create delivery step at order destination
            if step_count < MAX_STEPS_PER_VEHICLE - 1:
                delivery_items = []
                for wh_node, pickup_data in pickups_by_warehouse.items():
                    for item in pickup_data['items']:
                        delivery_items.append({
                            'order_id': order_id,
                            'sku_id': item['sku_id'],
                            'quantity': item['quantity']
                        })
                
                steps.append({
                    'node_id': order_node,
                    'pickups': [],
                    'deliveries': delivery_items,
                    'unloads': []
                })
                current_node = order_node
                step_count += 1
            
            processed_orders.add(order_id)
            log(f"Vehicle {vehicle_id} planned order {order_id}, remaining capacity: ({remaining_weight:.1f}kg, {remaining_volume:.1f}m³)")
            
            # Check if we should continue (capacity or complexity limits)
            if (remaining_weight < 1.0 or remaining_volume < 0.01 or 
                step_count >= MAX_STEPS_PER_VEHICLE - 1):
                break
        
        # Return to home
        if current_node != home_node:
            steps.append({'node_id': home_node, 'pickups': [], 'deliveries': [], 'unloads': []})
        
        log(f"Vehicle {vehicle_id} route completed with {len(steps)} steps, processed {len(processed_orders)} orders")
        return steps
        
    except Exception as e:
        log(f"Error planning route for vehicle {vehicle_id}: {e}")
        # Fallback: minimal home->home route
        try:
            home_node = env.get_vehicle_home_warehouse(vehicle_id)
            return [{'node_id': home_node, 'pickups': [], 'deliveries': [], 'unloads': []}]
        except:
            return []
 
 
def solver(env) -> Dict:
    """
    Main solver function for Robin Logistics Environment.
 
    Args:
        env: LogisticsEnvironment instance
 
    Returns:
        Complete solution dict with routes and sequential steps
    """
    log("Starting MVP solver")
    
    try:
        # 1. Data intake
        order_ids = env.get_all_order_ids()
        vehicle_ids = env.get_available_vehicles()
        
        log(f"Found {len(order_ids)} orders and {len(vehicle_ids)} vehicles")
        
        if not order_ids or not vehicle_ids:
            log("No orders or vehicles available")
            return {"routes": []}
        
        # Build shadow inventory
        shadow_inventory = {}
        for wh_id, warehouse in env.warehouses.items():
            try:
                inventory = env.get_warehouse_inventory(wh_id)
                shadow_inventory[wh_id] = dict(inventory) if inventory else {}
            except Exception:
                shadow_inventory[wh_id] = {}
        
        log(f"Built shadow inventory for {len(shadow_inventory)} warehouses")
        
        # Prepare order data with scoring
        orders_data = []
        for order_id in order_ids:
            try:
                requirements = env.get_order_requirements(order_id)
                destination = env.get_order_location(order_id)
                if requirements and destination is not None:
                    score = compute_order_score(env, order_id, shadow_inventory)
                    orders_data.append({
                        'id': order_id,
                        'requirements': requirements,
                        'destination': destination,
                        'score': score
                    })
            except Exception as e:
                log(f"Error processing order {order_id}: {e}")
                continue
        
        # Sort orders by priority (lower score = higher priority)
        orders_data.sort(key=lambda x: (x['score'], x['id']))
        log(f"Prioritized {len(orders_data)} orders")
        
        # 2. Generate routes for each vehicle
        routes = []
        for vehicle_id in vehicle_ids:
            try:
                steps = plan_vehicle_route(env, vehicle_id, orders_data, shadow_inventory)
                
                if len(steps) > 1:  # More than just home step
                    # Validate route steps
                    is_valid, message = env.validator.validate_route_steps(vehicle_id, steps)
                    
                    if not is_valid:
                        log(f"Route validation failed for {vehicle_id}: {message}")
                        # Fallback: try removing last order segment
                        if len(steps) > 3:  # At least home + pickup + delivery + home
                            # Remove last delivery and preceding pickups
                            fallback_steps = [steps[0]]  # Keep start
                            for step in steps[1:-2]:  # Skip last delivery and return
                                fallback_steps.append(step)
                            fallback_steps.append(steps[-1])  # Keep return home
                            
                            is_valid, message = env.validator.validate_route_steps(vehicle_id, fallback_steps)
                            if is_valid:
                                steps = fallback_steps
                                log(f"Fallback route valid for {vehicle_id}")
                            else:
                                # Ultimate fallback: home only
                                home_node = env.get_vehicle_home_warehouse(vehicle_id)
                                steps = [{'node_id': home_node, 'pickups': [], 'deliveries': [], 'unloads': []}]
                                log(f"Using minimal route for {vehicle_id}")
                        else:
                            # Ultimate fallback
                            try:
                                home_node = env.get_vehicle_home_warehouse(vehicle_id)
                                steps = [{'node_id': home_node, 'pickups': [], 'deliveries': [], 'unloads': []}]
                            except:
                                steps = []
                    
                    if steps:
                        routes.append({
                            'vehicle_id': vehicle_id,
                            'steps': steps
                        })
                        
            except Exception as e:
                log(f"Error generating route for vehicle {vehicle_id}: {e}")
                continue
        
        # 3. Finalize solution
        solution = {"routes": routes}
        
        # Final validation
        try:
            is_valid, message = env.validate_solution_complete(solution)
            if not is_valid:
                log(f"Solution validation failed: {message}")
                # Strip empty routes and try again
                filtered_routes = [r for r in routes if len(r.get('steps', [])) > 1]
                solution = {"routes": filtered_routes}
                
                is_valid, message = env.validate_solution_complete(solution)
                if not is_valid:
                    log(f"Filtered solution still invalid: {message}")
        except Exception as e:
            log(f"Error in final validation: {e}")
        
        log(f"Solution generated with {len(solution['routes'])} routes")
    return solution
        
    except Exception as e:
        log(f"Critical error in solver: {e}")
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

 
if __name__ == '__main__':
    # Test mode - requires environment to be available
    try:
        from robin_logistics import LogisticsEnvironment
        env = LogisticsEnvironment()
        solution = solver(env)
        print(f"Generated solution with {len(solution.get('routes', []))} routes")
    except ImportError:
        print("Robin Logistics Environment not available for standalone testing")
    except Exception as e:
        print(f"Error in test mode: {e}")