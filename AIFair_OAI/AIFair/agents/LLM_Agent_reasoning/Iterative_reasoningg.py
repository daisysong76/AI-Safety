import os
import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import wandb
from tqdm import tqdm
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

@dataclass
class ReasoningStep:
    prompt: str
    cot_candidates: List[str]
    preferred_index: Optional[int] = None
    step_scores: Optional[List[float]] = None

class CoTDataset(Dataset):
    def __init__(self, reasoning_steps: List[ReasoningStep]):
        self.reasoning_steps = reasoning_steps
    
    def __len__(self):
        return len(self.reasoning_steps)
    
    def __getitem__(self, idx):
        return self.reasoning_steps[idx]

class IRPOTrainer:
    def __init__(self, config: Dict):
        """
        Initialize IRPO Trainer with configuration.
        
        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.device = config['device']
        
        # Initialize model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
        self.model = AutoModelForCausalLM.from_pretrained(config['base_model'])
        self.model.to(self.device)
        
        # Optimizer setup
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Logging setup
        self.setup_logging()
        
        # Training setup
        self.max_length = config.get('max_length', 512)
        self.num_iterations = config.get('num_iterations', 3)
        self.batch_size = config.get('batch_size', 8)
        
    def setup_logging(self):
        """Setup WandB and TensorBoard logging"""
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'irpo-trainer'),
                name=self.config.get('wandb_run_name', 'irpo_run'),
                config=self.config
            )
        
        if self.config.get('use_tensorboard', False):
            self.tensorboard = SummaryWriter(
                log_dir=self.config.get('tensorboard_dir', './runs/irpo')
            )
        else:
            self.tensorboard = None
    
    def generate_cot_candidates(self, prompt: str, num_candidates: int = 3) -> List[str]:
        """Generate multiple Chain-of-Thought candidates for a given prompt"""
        candidates = []
        
        for _ in range(num_candidates):
            inputs = self.tokenizer(
                prompt + "\nLet's solve this step by step:",
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            # Generate with some randomness for diversity
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                do_sample=True,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
            candidate = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            candidates.append(candidate)
        
        return candidates
    
    def compute_preference_loss(
        self,
        preferred: torch.Tensor,
        other: torch.Tensor,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Compute preference loss between two candidates"""
        logits_diff = (preferred - other) / temperature
        loss = -F.logsigmoid(logits_diff)
        return loss.mean()
    
    def train_step(self, reasoning_step: ReasoningStep) -> Dict[str, float]:
        """Perform single training step with preference optimization"""
        self.optimizer.zero_grad()
        
        # Get logits for all candidates
        candidate_logits = []
        for cot in reasoning_step.cot_candidates:
            inputs = self.tokenizer(
                cot,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            outputs = self.model(**inputs)
            candidate_logits.append(outputs.logits.mean())
        
        # Compute preference loss if preferred index is available
        losses = []
        if reasoning_step.preferred_index is not None:
            preferred_logits = candidate_logits[reasoning_step.preferred_index]
            
            for i, other_logits in enumerate(candidate_logits):
                if i != reasoning_step.preferred_index:
                    loss = self.compute_preference_loss(
                        preferred_logits,
                        other_logits,
                        temperature=self.config.get('temperature', 1.0)
                    )
                    losses.append(loss)
        
        # Compute total loss and backward
        if losses:
            total_loss = sum(losses) / len(losses)
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('max_grad_norm', 1.0)
            )
            
            self.optimizer.step()
            
            return {"loss": total_loss.item()}
        
        return {"loss": 0.0}
    
    def evaluate_candidates(
        self,
        candidates: List[str],
        reference: Optional[str] = None
    ) -> List[float]:
        """Evaluate candidates using model scores and optional reference"""
        scores = []
        
        for candidate in candidates:
            inputs = self.tokenizer(
                candidate,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = outputs.logits.mean().item()
                
                # If reference is provided, include similarity score
                if reference:
                    similarity = self.compute_similarity(candidate, reference)
                    score = 0.7 * score + 0.3 * similarity
                
                scores.append(score)
        
        return scores
    
    def train_iteration(
        self,
        dataset: CoTDataset,
        iteration: int
    ) -> Dict[str, float]:
        """Run one complete iteration of IRPO training"""
        self.model.train()
        total_loss = 0
        steps = 0
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        for batch in tqdm(dataloader, desc=f"Iteration {iteration + 1}"):
            for reasoning_step in batch:
                # Generate new candidates if needed
                if len(reasoning_step.cot_candidates) < self.config.get('num_candidates', 3):
                    new_candidates = self.generate_cot_candidates(
                        reasoning_step.prompt,
                        self.config.get('num_candidates', 3) - len(reasoning_step.cot_candidates)
                    )
                    reasoning_step.cot_candidates.extend(new_candidates)
                
                # Evaluate and update preferences
                scores = self.evaluate_candidates(reasoning_step.cot_candidates)
                reasoning_step.preferred_index = np.argmax(scores)
                reasoning_step.step_scores = scores
                
                # Train step
                metrics = self.train_step(reasoning_step)
                total_loss += metrics["loss"]
                steps += 1
        
        avg_loss = total_loss / steps
        
        # Logging
        if self.config.get('use_wandb', False):
            wandb.log({
                'iteration': iteration + 1,
                'avg_loss': avg_loss
            })
        
        if self.tensorboard:
            self.tensorboard.add_scalar('Loss/train', avg_loss, iteration)
        
        return {"avg_loss": avg_loss}
    
    def train(self, dataset: CoTDataset):
        """Run complete IRPO training process"""
        for iteration in range(self.num_iterations):
            metrics = self.train_iteration(dataset, iteration)
            
            print(f"Iteration {iteration + 1}/{self.num_iterations}")
            print(f"Average Loss: {metrics['avg_loss']:.4f}")
        
        # Save final model
        if self.config.get('output_dir'):
            os.makedirs(self.config['output_dir'], exist_ok=True)
            self.model.save_pretrained(self.config['output_dir'])
            self.tokenizer.save_pretrained(self.config['output_dir'])
        
        # Clean up logging
        if self.tensorboard:
            self.tensorboard.close()
        if self.config.get('use_wandb', False):
            wandb.finish()

# Example usage
if __name__ == "__main__":
    config = {
        'base_model': 'gpt2-medium',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'learning_rate': 2e-5,
        'num_iterations': 3,
        'batch_size': 8,
        'max_length': 512,
        'num_candidates': 3,
        'temperature': 0.7,
        'use_wandb': True,
        'wandb_project': 'irpo-trainer',
        'output_dir': './irpo_model',
        'use_tensorboard': True,
        'tensorboard_dir': './runs/irpo'
    }
    
    # Create sample dataset
    reasoning_steps = [
        ReasoningStep(
            prompt="What is the sum of the first 10 natural numbers?",
            cot_candidates=[
                "Let's solve step by step:\n1. Write out numbers: 1,2,3,4,5,6,7,8,9,10\n2. Use formula: (n*(n+1))/2\n3. n=10, so: (10*11)/2 = 55\nAnswer: 55"
            ]
        )
        # Add more reasoning steps...
    ]
    
    dataset = CoTDataset(reasoning_steps)
    trainer = IRPOTrainer(config)
    trainer.train(dataset)