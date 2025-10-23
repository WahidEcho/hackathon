#!/usr/bin/env python3
"""
RL Training System for Robin Logistics MWVRP
Trains Deep Q-Learning model using Robin Logistics Environment
"""

import numpy as np
import time
import os
from typing import Dict, List, Tuple
from rl_model import LogisticsRLModel


class RLTrainer:
    """
    Reinforcement Learning trainer for logistics optimization.
    """
    
    def __init__(self, model: LogisticsRLModel):
        self.model = model
        self.training_history = []
        
    def train_episode(self, env) -> float:
        """
        Train one episode.
        
        Args:
            env: LogisticsEnvironment instance
            
        Returns:
            Episode reward
        """
        # Reset environment state (generate new scenario)
        try:
            # Get initial state
            state = self.model.encode_state(env)
            episode_reward = 0.0
            max_steps = 50  # Limit episode length
            
            for step in range(max_steps):
                # Get action from policy
                action = self.model.get_action(state, exploration=True)
                
                # Convert action to solution
                solution = self.model.decode_action_to_solution(action, env)
                
                # Calculate reward
                reward = self.model.calculate_reward(solution, env)
                episode_reward += reward
                
                # Get next state (simplified - could be more sophisticated)
                next_state = self.model.encode_state(env)
                
                # Store experience in memory
                experience = (state.copy(), action, reward, next_state.copy(), False)  # Not done
                self.model.memory.append(experience)
                
                # Limit memory size
                if len(self.model.memory) > self.model.max_memory:
                    self.model.memory.pop(0)
                
                # Update state
                state = next_state
                
                # Break if we have a good solution
                if reward > 1000:
                    break
            
            # Update model with experience replay
            self.update_model()
            
            return episode_reward
            
        except Exception as e:
            print(f"Episode training error: {e}")
            return -1000.0
    
    def update_model(self):
        """Update model using experience replay."""
        if len(self.model.memory) < 32:  # Minimum batch size
            return
        
        # Sample random batch from memory
        batch_size = min(32, len(self.model.memory))
        batch_indices = np.random.choice(len(self.model.memory), batch_size, replace=False)
        
        states = []
        actions = []
        rewards = []
        next_states = []
        
        for idx in batch_indices:
            state, action, reward, next_state, done = self.model.memory[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
        
        states = np.array(states)
        next_states = np.array(next_states)
        
        # Calculate target Q-values
        current_q = np.array([self.model.forward(state) for state in states])
        next_q = np.array([self.model.forward(state) for state in next_states])
        
        # Update Q-values using Bellman equation
        for i in range(batch_size):
            target = rewards[i] + self.model.gamma * np.max(next_q[i])
            current_q[i][actions[i]] = target
        
        # Simple gradient update (simplified - normally would use backpropagation)
        self.simple_weight_update(states, current_q)
    
    def simple_weight_update(self, states: np.ndarray, target_q: np.ndarray):
        """Simplified weight update (approximation of gradient descent)."""
        learning_rate = self.model.learning_rate
        
        # Calculate prediction error
        predicted_q = np.array([self.model.forward(state) for state in states])
        error = target_q - predicted_q
        
        # Update weights (simplified approach)
        try:
            # Update output layer weights
            for i in range(len(states)):
                state = states[i]
                h1 = self.model.relu(np.dot(state, self.model.weights['w1']) + self.model.weights['b1'])
                h2 = self.model.relu(np.dot(h1, self.model.weights['w2']) + self.model.weights['b2'])
                
                # Update w3 and b3
                self.model.weights['w3'] += learning_rate * np.outer(h2, error[i]) / len(states)
                self.model.weights['b3'] += learning_rate * error[i] / len(states)
                
                # Update w2 and b2 (simplified)
                error_h2 = np.dot(error[i], self.model.weights['w3'].T)
                error_h2[h2 <= 0] = 0  # ReLU derivative
                
                self.model.weights['w2'] += learning_rate * np.outer(h1, error_h2) / len(states)
                self.model.weights['b2'] += learning_rate * error_h2 / len(states)
                
        except Exception as e:
            pass  # Skip update if error occurs
    
    def train(self, num_episodes: int = 100, save_interval: int = 20):
        """
        Train the RL model.
        
        Args:
            num_episodes: Number of training episodes
            save_interval: Save model every N episodes
        """
        print(f"🚀 Starting RL Training for {num_episodes} episodes")
        print("=" * 60)
        
        try:
            # Import environment
            from robin_logistics import LogisticsEnvironment
            
            best_reward = float('-inf')
            recent_rewards = []
            
            for episode in range(num_episodes):
                # Create fresh environment for each episode
                env = LogisticsEnvironment()
                
                # Generate new scenario with some randomization
                if episode % 5 == 0:  # Change scenario every 5 episodes
                    try:
                        env.generate_new_scenario(seed=episode)
                    except:
                        pass  # Use default scenario if generation fails
                
                # Train one episode
                episode_reward = self.train_episode(env)
                recent_rewards.append(episode_reward)
                
                # Keep only recent rewards for averaging
                if len(recent_rewards) > 10:
                    recent_rewards.pop(0)
                
                # Update model statistics
                self.model.episodes_trained += 1
                self.model.total_reward += episode_reward
                
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    self.model.best_reward = best_reward
                
                # Log progress
                if episode % 10 == 0 or episode == num_episodes - 1:
                    avg_reward = np.mean(recent_rewards)
                    print(f"Episode {episode+1:3d}/{num_episodes}: "
                          f"Reward: {episode_reward:8.1f} | "
                          f"Avg: {avg_reward:8.1f} | "
                          f"Best: {best_reward:8.1f} | "
                          f"Epsilon: {self.model.epsilon:.3f}")
                
                # Save model periodically
                if (episode + 1) % save_interval == 0:
                    self.model.save_model('rl_model_checkpoint.pkl')
                    print(f"📁 Model saved at episode {episode + 1}")
                
                # Decay exploration rate
                if self.model.epsilon > 0.01:
                    self.model.epsilon *= 0.995
                
                # Add training history
                self.training_history.append({
                    'episode': episode + 1,
                    'reward': episode_reward,
                    'avg_reward': np.mean(recent_rewards),
                    'best_reward': best_reward,
                    'epsilon': self.model.epsilon
                })
            
            # Final save
            self.model.save_model('trained_model.pkl')
            print(f"\n✅ Training completed! Model saved as 'trained_model.pkl'")
            print(f"📊 Final Stats:")
            print(f"   Episodes: {self.model.episodes_trained}")
            print(f"   Best Reward: {self.model.best_reward:.1f}")
            print(f"   Final Epsilon: {self.model.epsilon:.3f}")
            
            return True
            
        except ImportError:
            print("❌ robin-logistics-env not found. Please install: pip install robin-logistics-env")
            return False
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False
    
    def evaluate_model(self, num_episodes: int = 10) -> Dict:
        """Evaluate the trained model."""
        print(f"🧪 Evaluating model over {num_episodes} episodes...")
        
        try:
            from robin_logistics import LogisticsEnvironment
            
            rewards = []
            fulfillments = []
            costs = []
            
            # Disable exploration for evaluation
            original_epsilon = self.model.epsilon
            self.model.epsilon = 0.0
            
            for episode in range(num_episodes):
                env = LogisticsEnvironment()
                
                # Generate test scenario
                if episode < 5:
                    env.generate_new_scenario(seed=42 + episode)
                
                state = self.model.encode_state(env)
                action = self.model.get_action(state, exploration=False)
                solution = self.model.decode_action_to_solution(action, env)
                
                # Calculate metrics
                reward = self.model.calculate_reward(solution, env)
                rewards.append(reward)
                
                # Calculate fulfillment
                total_deliveries = sum(len(step.get('deliveries', [])) for route in solution.get('routes', []) for step in route.get('steps', []))
                fulfillments.append(total_deliveries)
                
                # Calculate cost
                try:
                    cost = env.calculate_solution_cost(solution)
                    costs.append(cost)
                except:
                    costs.append(0)
            
            # Restore exploration
            self.model.epsilon = original_epsilon
            
            # Calculate statistics
            results = {
                'avg_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'avg_fulfillment': np.mean(fulfillments),
                'avg_cost': np.mean(costs),
                'success_rate': sum(1 for r in rewards if r > 0) / len(rewards)
            }
            
            print(f"📊 Evaluation Results:")
            print(f"   Average Reward: {results['avg_reward']:.1f} ± {results['std_reward']:.1f}")
            print(f"   Average Deliveries: {results['avg_fulfillment']:.1f}")
            print(f"   Average Cost: ${results['avg_cost']:.2f}")
            print(f"   Success Rate: {results['success_rate']:.1%}")
            
            return results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return {}


def main():
    """Main training function."""
    print("🧠 Robin Logistics RL Training System")
    print("=" * 50)
    
    # Initialize model
    model = LogisticsRLModel(state_dim=100, action_dim=50, learning_rate=0.001)
    trainer = RLTrainer(model)
    
    # Train model
    success = trainer.train(num_episodes=200, save_interval=25)
    
    if success:
        # Evaluate trained model
        trainer.evaluate_model(num_episodes=10)
        
        print(f"\n🎯 RL Training Complete!")
        print(f"📁 Trained model saved as 'trained_model.pkl'")
        print(f"💡 Use this model in solver.py for inference")
        
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
