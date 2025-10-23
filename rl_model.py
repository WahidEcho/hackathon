#!/usr/bin/env python3
"""
Deep Q-Learning Model for Robin Logistics MWVRP
Reinforcement Learning approach for vehicle routing optimization
"""

import numpy as np
import json
import pickle
import os
from typing import Dict, List, Tuple, Any, Optional


class LogisticsRLModel:
    """
    Deep Q-Learning model for logistics vehicle routing.
    Uses neural network approximation for Q-values.
    """
    
    def __init__(self, state_dim: int = 100, action_dim: int = 50, learning_rate: float = 0.001):
        """Initialize the RL model."""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Simple neural network approximation (using basic math, no external ML libs)
        self.weights = {
            'w1': np.random.normal(0, 0.1, (state_dim, 64)),
            'b1': np.zeros(64),
            'w2': np.random.normal(0, 0.1, (64, 32)),
            'b2': np.zeros(32),
            'w3': np.random.normal(0, 0.1, (32, action_dim)),
            'b3': np.zeros(action_dim)
        }
        
        # Q-learning parameters
        self.epsilon = 0.1  # Exploration rate
        self.gamma = 0.95   # Discount factor
        self.memory = []    # Experience replay buffer
        self.max_memory = 10000
        
        # Training statistics
        self.episodes_trained = 0
        self.total_reward = 0.0
        self.best_reward = float('-inf')
        
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        h1 = self.relu(np.dot(state, self.weights['w1']) + self.weights['b1'])
        h2 = self.relu(np.dot(h1, self.weights['w2']) + self.weights['b2'])
        q_values = np.dot(h2, self.weights['w3']) + self.weights['b3']
        return q_values
    
    def get_action(self, state: np.ndarray, exploration: bool = True) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: Current state representation
            exploration: Whether to use exploration (epsilon-greedy)
            
        Returns:
            Action index
        """
        if exploration and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        
        q_values = self.forward(state)
        return np.argmax(q_values)
    
    def encode_state(self, env) -> np.ndarray:
        """
        Encode environment state into fixed-size vector.
        
        Args:
            env: LogisticsEnvironment instance
            
        Returns:
            State vector of size state_dim
        """
        state = np.zeros(self.state_dim)
        
        try:
            # Basic scenario information
            orders = env.get_all_order_ids()
            vehicles = env.get_available_vehicles()
            warehouses = env.warehouses
            
            # Encode basic counts (first 10 features)
            state[0] = len(orders) / 100.0  # Normalize
            state[1] = len(vehicles) / 20.0
            state[2] = len(warehouses) / 10.0
            
            # Encode vehicle capacities (features 3-12)
            for i, vehicle_id in enumerate(vehicles[:10]):
                try:
                    remaining_weight, remaining_volume = env.get_vehicle_remaining_capacity(vehicle_id)
                    state[3 + i] = remaining_weight / 5000.0  # Normalize
                    if i < 5:  # Volume for first 5 vehicles
                        state[8 + i] = remaining_volume / 20.0
                except:
                    pass
            
            # Encode warehouse inventory levels (features 13-32)
            inventory_features = []
            for wh_id, wh in warehouses.items():
                try:
                    inventory = env.get_warehouse_inventory(wh_id)
                    total_inventory = sum(inventory.values()) if inventory else 0
                    inventory_features.append(total_inventory)
                except:
                    inventory_features.append(0)
            
            for i, inv in enumerate(inventory_features[:20]):
                state[13 + i] = inv / 1000.0  # Normalize
            
            # Encode order requirements (features 33-62)
            order_features = []
            for order_id in orders[:30]:
                try:
                    requirements = env.get_order_requirements(order_id)
                    total_items = sum(requirements.values()) if requirements else 0
                    order_features.append(total_items)
                except:
                    order_features.append(0)
            
            for i, req in enumerate(order_features):
                if 33 + i < len(state):
                    state[33 + i] = req / 100.0
            
            # Add some randomness for exploration (features 63-99)
            state[63:] = np.random.normal(0, 0.01, len(state) - 63)
            
        except Exception as e:
            # Fallback: random state
            state = np.random.normal(0, 0.1, self.state_dim)
            
        return state
    
    def decode_action_to_solution(self, action: int, env) -> Dict:
        """
        Convert RL action to solution format.
        
        Args:
            action: Action index from RL model
            env: LogisticsEnvironment instance
            
        Returns:
            Solution dictionary
        """
        try:
            # Simple action decoding strategy
            orders = env.get_all_order_ids()
            vehicles = env.get_available_vehicles()
            
            if not orders or not vehicles:
                return {"routes": []}
            
            # Use action to determine assignment strategy
            assignment_strategy = action % 3  # 0: round-robin, 1: capacity-based, 2: distance-based
            vehicle_selection = (action // 3) % len(vehicles)
            order_prioritization = (action // 9) % 5
            
            routes = []
            
            # Distribute orders among vehicles based on strategy
            for i, vehicle_id in enumerate(vehicles):
                try:
                    home_node = env.get_vehicle_home_warehouse(vehicle_id)
                    if home_node is None:
                        continue
                    
                    # Simple route: home -> pickup -> deliver -> home
                    steps = [{'node_id': home_node, 'pickups': [], 'deliveries': [], 'unloads': []}]
                    
                    # Select orders for this vehicle based on action
                    vehicle_orders = []
                    if assignment_strategy == 0:  # Round-robin
                        for j in range(i, len(orders), len(vehicles)):
                            if j < len(orders):
                                vehicle_orders.append(orders[j])
                    elif assignment_strategy == 1:  # Capacity-based
                        vehicle_orders = orders[i*2:(i+1)*2]  # 2 orders per vehicle
                    else:  # Distance-based (simplified)
                        vehicle_orders = orders[i:i+3]  # 3 orders per vehicle
                    
                    # Generate pickups and deliveries
                    pickup_items = []
                    delivery_items = []
                    
                    for order_id in vehicle_orders[:3]:  # Limit to 3 orders
                        if order_id in orders:
                            try:
                                requirements = env.get_order_requirements(order_id)
                                destination = env.get_order_location(order_id)
                                
                                if requirements and destination:
                                    # Find warehouse with items
                                    for wh_id, wh in env.warehouses.items():
                                        if wh.location and wh.location.id == home_node:
                                            for sku_id, qty in requirements.items():
                                                pickup_items.append({
                                                    'warehouse_id': wh_id,
                                                    'sku_id': sku_id,
                                                    'quantity': min(qty, 5)  # Limit quantity
                                                })
                                                delivery_items.append({
                                                    'order_id': order_id,
                                                    'sku_id': sku_id,
                                                    'quantity': min(qty, 5)
                                                })
                                            break
                                    
                                    # Add delivery step
                                    if delivery_items:
                                        steps.append({
                                            'node_id': destination,
                                            'pickups': [],
                                            'deliveries': [d for d in delivery_items if d['order_id'] == order_id],
                                            'unloads': []
                                        })
                            except:
                                continue
                    
                    # Add pickup step if we have items
                    if pickup_items:
                        steps.insert(1, {
                            'node_id': home_node,
                            'pickups': pickup_items,
                            'deliveries': [],
                            'unloads': []
                        })
                    
                    # Return home
                    if len(steps) > 1 and steps[-1]['node_id'] != home_node:
                        steps.append({'node_id': home_node, 'pickups': [], 'deliveries': [], 'unloads': []})
                    
                    if len(steps) > 1:
                        routes.append({
                            'vehicle_id': vehicle_id,
                            'steps': steps
                        })
                        
                except Exception:
                    continue
            
            return {"routes": routes}
            
        except Exception:
            return {"routes": []}
    
    def calculate_reward(self, solution: Dict, env) -> float:
        """
        Calculate reward based on solution quality.
        Uses the reward function from the project description.
        """
        try:
            # Count operations
            total_pickups = sum(len(step.get('pickups', [])) for route in solution.get('routes', []) for step in route.get('steps', []))
            total_deliveries = sum(len(step.get('deliveries', [])) for route in solution.get('routes', []) for step in route.get('steps', []))
            
            # Basic reward components
            W_DELIVER = 200.0
            ALPHA_DIST = 1.0
            W_UTIL = 0.1
            W_VIOLATION = 500.0
            
            # Reward for deliveries
            reward = W_DELIVER * total_deliveries
            
            # Penalty for route length (simplified)
            total_steps = sum(len(route.get('steps', [])) for route in solution.get('routes', []))
            reward -= ALPHA_DIST * total_steps
            
            # Bonus for utilization (simplified)
            if len(solution.get('routes', [])) > 0:
                avg_utilization = total_pickups / len(solution.get('routes', []))
                reward += W_UTIL * avg_utilization
            
            # Terminal bonus for fulfillment
            if total_deliveries > 100:
                reward += 500.0  # B_FULFILL bonus
            
            return reward
            
        except Exception:
            return -1000.0  # Penalty for invalid solutions
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        model_data = {
            'weights': self.weights,
            'episodes_trained': self.episodes_trained,
            'total_reward': self.total_reward,
            'best_reward': self.best_reward,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'epsilon': self.epsilon,
            'gamma': self.gamma
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model."""
        try:
            if not os.path.exists(filepath):
                return False
                
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.weights = model_data['weights']
            self.episodes_trained = model_data.get('episodes_trained', 0)
            self.total_reward = model_data.get('total_reward', 0.0)
            self.best_reward = model_data.get('best_reward', float('-inf'))
            self.state_dim = model_data.get('state_dim', self.state_dim)
            self.action_dim = model_data.get('action_dim', self.action_dim)
            self.epsilon = model_data.get('epsilon', 0.1)
            self.gamma = model_data.get('gamma', 0.95)
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model."""
        return {
            'episodes_trained': self.episodes_trained,
            'total_reward': self.total_reward,
            'best_reward': self.best_reward,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'model_loaded': True
        }
