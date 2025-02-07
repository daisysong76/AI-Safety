import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class ReasoningStep:
    """Represents a single step in the reasoning process"""
    text: str
    intermediate_output: torch.Tensor
    target_score: float

class ProcessRewardModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 3
    ):
        super().__init__()
        
        # Encoder for processing text at each step
        self.step_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                nhead=8,
                dim_feedforward=hidden_dim
            ),
            num_layers=num_layers
        )
        
        # Reward prediction head
        self.reward_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(
        self,
        reasoning_steps: List[ReasoningStep]
    ) -> Tuple[torch.Tensor, List[float]]:
        """
        Process a sequence of reasoning steps and predict rewards
        Returns both step-wise and final rewards
        """
        step_embeddings = []
        step_rewards = []
        
        # Process each reasoning step
        for step in reasoning_steps:
            # Encode the step's text and intermediate output
            step_embed = self.step_encoder(step.intermediate_output)
            step_embeddings.append(step_embed)
            
            # Predict reward for this step
            step_reward = self.reward_head(step_embed)
            step_rewards.append(step_reward.item())
            
        # Combine step embeddings for final prediction
        sequence_embed = torch.stack(step_embeddings)
        final_reward = self.reward_head(sequence_embed.mean(dim=0))
        
        return final_reward, step_rewards

class PRMTrainer:
    def __init__(
        self,
        model: ProcessRewardModel,
        lr: float = 1e-4
    ):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    def train_step(
        self,
        reasoning_steps: List[ReasoningStep]
    ) -> Dict[str, float]:
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Get model predictions
        final_reward, step_rewards = self.model(reasoning_steps)
        
        # Calculate losses
        step_loss = 0
        for pred, step in zip(step_rewards, reasoning_steps):
            step_loss += F.mse_loss(
                torch.tensor(pred),
                torch.tensor(step.target_score)
            )
        
        # Final trajectory reward loss
        final_target = torch.tensor(
            sum(step.target_score for step in reasoning_steps) / len(reasoning_steps)
        )
        final_loss = F.mse_loss(final_reward.squeeze(), final_target)
        
        # Combined loss
        total_loss = step_loss + final_loss
        total_loss.backward()
        
        self.optimizer.step()
        
        return {
            "step_loss": step_loss.item(),
            "final_loss": final_loss.item(),
            "total_loss": total_loss.item()
        }

# Example usage
def train_prm(
    model: ProcessRewardModel,
    training_data: List[List[ReasoningStep]],
    num_epochs: int = 10
):
    trainer = PRMTrainer(model)
    
    for epoch in range(num_epochs):
        epoch_losses = []
        for reasoning_sequence in training_data:
            losses = trainer.train_step(reasoning_sequence)
            epoch_losses.append(losses["total_loss"])
        
        print(f"Epoch {epoch+1}, Avg Loss: {sum(epoch_losses)/len(epoch_losses):.4f}")

# Here's a breakdown of how this implementation works:

# Model Architecture:

# Uses a transformer encoder to process each reasoning step
# Includes a reward prediction head to score both individual steps and the full trajectory
# Maintains state across reasoning steps to evaluate the full process


# Training Process:

# Takes sequences of reasoning steps as input
# Each step includes the text, intermediate outputs, and target scores
# Computes both step-wise and final trajectory rewards
# Uses MSE loss to train against human-provided scores


# Key Features:

# Step-by-step reward prediction
# Combined loss function balancing individual steps and final outcomes
# Flexible architecture that can handle variable-length reasoning chains



# To use this implementation, you would need to:

# Prepare your training data as sequences of ReasoningStep objects
# Initialize the model with appropriate dimensions
# Run the training loop with your data