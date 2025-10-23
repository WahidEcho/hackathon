#!/usr/bin/env python3
"""
Quick RL Model Training Script
Generates a basic trained model for immediate use
"""

import sys
import os
from rl_trainer import RLTrainer
from rl_model import LogisticsRLModel


def main():
    """Quick training to generate a working model."""
    print("🚀 Quick RL Model Training")
    print("=" * 40)
    
    # Check if model already exists
    if os.path.exists('trained_model.pkl'):
        print("✅ Trained model already exists!")
        print("💡 Delete 'trained_model.pkl' to retrain")
        return 0
    
    try:
        # Initialize model and trainer
        model = LogisticsRLModel(state_dim=100, action_dim=50, learning_rate=0.01)
        trainer = RLTrainer(model)
        
        print("🧠 Training RL model (quick version)...")
        
        # Quick training session
        success = trainer.train(num_episodes=50, save_interval=10)
        
        if success:
            print("✅ Quick training completed!")
            print("📁 Model saved as 'trained_model.pkl'")
            print("🎯 Ready for use with solver.py")
            return 0
        else:
            print("❌ Training failed")
            return 1
            
    except Exception as e:
        print(f"❌ Training error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
