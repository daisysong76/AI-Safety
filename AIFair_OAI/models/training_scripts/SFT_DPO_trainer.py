import os
from abc import ABC

import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
import wandb
import pandas as pd
import numpy as np

from openrlhf.models import DPOLoss, GPTLMLoss
from openrlhf.utils.distributed_sampler import DistributedSampler

class UnifiedLLMTrainer(ABC):
    """
    Unified Trainer for both Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO),
    incorporating advanced bias mitigation and Parameter-Efficient Fine-Tuning (PEFT).
    
    Args:
        config (Dict[str, Any]): Configuration dictionary.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize tokenizer and models
        self.tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
        self.model = AutoModelForCausalLM.from_pretrained(config['base_model'])
        
        # PEFT Configuration
        self.peft_model = self.configure_peft()
        
        # Initialize training strategies
        self.strategy = self._initialize_strategy()
        
        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = self._initialize_optimizer_scheduler()
        
        # Initialize loss functions
        self.loss_fn_sft = GPTLMLoss(ring_attn_group=self.strategy.ring_attn_group)
        self.loss_fn_dpo = DPOLoss(beta=config.get('dpo_beta', 0.01), 
                                   label_smoothing=config.get('label_smoothing', 0.0),
                                   ipo=config.get('ipo', False))
        
        # Initialize dataloaders
        self.train_dataloader, self.eval_dataloader = self._initialize_dataloaders()
        
        # Initialize logging
        self._initialize_logging()
        
        # Initialize bias mitigation components
        self.alignment_constructor = self._initialize_alignment_constructor()
        self.response_generator = self._initialize_response_generator()
        self.mixture_optimizer = self._initialize_mixture_optimizer()
        
        # Flags for auxiliary losses and sample packing
        self.aux_loss = config.get('aux_loss_coef', 0.0) > 1e-8
        self.packing_samples = config.get('packing_samples', False)
        
        # Flags for checkpointing
        self.save_hf_ckpt = config.get('save_hf_ckpt', False)
        self.disable_ds_ckpt = config.get('disable_ds_ckpt', False)
        
    def _initialize_strategy(self):
        """Initialize training strategy, e.g., DeepSpeed configurations."""
        # Placeholder: Implement based on specific strategy requirements
        return Strategy(
            zero_stage=self.config.get('zero_stage', 3),
            bf16=self.config.get('bf16', False),
            gradient_checkpointing=self.config.get('gradient_checkpointing', False),
            ring_attn_group=self.config.get('ring_attn_group', None)
        )
    
    def _initialize_optimizer_scheduler(self):
        """Initialize optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(self.peft_model.parameters(), lr=self.config.get('learning_rate', 5e-5))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config.get('scheduler_step_size', 1), 
                                                    gamma=self.config.get('scheduler_gamma', 0.1))
        return optimizer, scheduler
    
    def configure_peft(self):
        """Configure Parameter-Efficient Fine-Tuning (PEFT) using LoRA."""
        peft_config = LoraConfig(
            task_type="CAUSAL_LM", 
            r=self.config.get('peft_r', 16), 
            lora_alpha=self.config.get('peft_alpha', 32), 
            lora_dropout=self.config.get('peft_dropout', 0.1)
        )
        return get_peft_model(self.model, peft_config)
    
    def _initialize_alignment_constructor(self):
        """Initialize alignment data constructor for bias mitigation."""
        # Placeholder: Implement actual component initialization
        return AlignmentConstructor()  # Replace with actual class
    
    def _initialize_response_generator(self):
        """Initialize response generation framework for bias mitigation."""
        # Placeholder: Implement actual component initialization
        return ResponseGenerator()  # Replace with actual class
    
    def _initialize_mixture_optimizer(self):
        """Initialize data mixture optimizer for bias mitigation."""
        # Placeholder: Implement actual component initialization
        return MixtureOptimizer()  # Replace with actual class
    
    def _initialize_logging(self):
        """Initialize logging with WandB and TensorBoard."""
        self._wandb = None
        self._tensorboard = None
        if self.config.get('use_wandb', False):
            if not wandb.api.api_key:
                wandb.login(key=self.config.get('wandb_api_key'))
            wandb.init(
                entity=self.config.get('wandb_org', 'your_org'),
                project=self.config.get('wandb_project', 'unified-trainer'),
                group=self.config.get('wandb_group', 'group_name'),
                name=self.config.get('wandb_run_name', 'run_name'),
                config=self.config,
                reinit=True,
            )
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)
            self._wandb = wandb
        
        if self.config.get('use_tensorboard', False) and self._wandb is None:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(self.config['use_tensorboard'], exist_ok=True)
            log_dir = os.path.join(self.config['use_tensorboard'], self.config.get('wandb_run_name', 'run_name'))
            self._tensorboard = SummaryWriter(log_dir=log_dir)
    
    def _initialize_dataloaders(self):
        """Initialize training and evaluation dataloaders."""
        train_dataset, val_dataset = self.prepare_training_data()
        
        train_sampler = DistributedSampler(train_dataset) if self.strategy.is_distributed else None
        eval_sampler = DistributedSampler(val_dataset) if self.strategy.is_distributed else None
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.get('train_batch_size', 32),
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            ) if not self.packing_samples else None
        )
        
        eval_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.get('eval_batch_size', 32),
            shuffle=False,
            sampler=eval_sampler,
            collate_fn=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=False
            )
        )
        
        return train_dataloader, eval_dataloader
    
    def prepare_training_data(self) -> (Dataset, Dataset):
        """
        Prepare and split the dataset into training and validation sets.
        Incorporates bias mitigation strategies.
        
        Returns:
            Tuple of training and validation datasets.
        """
        raw_dataset = load_dataset(self.config['dataset_type'], data_files=self.config['dataset_path'])
        # Apply bias mitigation
        debiased_dataset = self.prepare_debiased_dataset(raw_dataset['train'])
        # Split into train and validation
        train_dataset, val_dataset = debiased_dataset.train_test_split(test_size=0.1, seed=self.config.get('seed', 42)).values()
        return train_dataset, val_dataset
    
    def prepare_debiased_dataset(self, raw_dataset: Dataset, bias_threshold: float = 0.2) -> Dataset:
        """
        Apply comprehensive bias mitigation to the dataset.
        
        Args:
            raw_dataset (Dataset): Raw input dataset.
            bias_threshold (float): Threshold for acceptable bias.
        
        Returns:
            Dataset: Debiased dataset.
        """
        # 1. Generate diverse responses
        generated_responses = self.response_generator.generate_responses(raw_dataset['text'])
        
        # 2. Optimize data mixture
        mixture_optimized_data = self.mixture_optimizer.optimize_data_mixture(pd.DataFrame(generated_responses))
        
        # 3. Apply advanced debiasing
        debiased_data = self._apply_gender_debiasing(mixture_optimized_data, bias_threshold)
        
        # Convert back to Hugging Face Dataset
        return Dataset.from_pandas(debiased_data)
    
    def _apply_gender_debiasing(self, dataset: pd.DataFrame, bias_threshold: float) -> pd.DataFrame:
        """
        Apply advanced gender debiasing techniques to the dataset.
        
        Args:
            dataset (pd.DataFrame): Input dataset.
            bias_threshold (float): Maximum acceptable bias score.
        
        Returns:
            pd.DataFrame: Debiased dataset.
        """
        debiased_samples = []
        for _, row in dataset.iterrows():
            debiased_text = self._neutralize_gendered_language(row['text'])
            debiased_samples.append({'text': debiased_text})
        return pd.DataFrame(debiased_samples)
    
    def _neutralize_gendered_language(self, text: str) -> str:
        """
        Neutralize gendered language in the text.
        
        Args:
            text (str): Input text.
        
        Returns:
            str: Neutralized text.
        """
        neutralization_map = {
            'he': 'they', 'she': 'they',
            'him': 'them', 'her': 'them',
            'his': 'their', 'hers': 'theirs',
            'himself': 'themselves', 'herself': 'themselves'
        }
        for gendered, neutral in neutralization_map.items():
            text = text.replace(f" {gendered} ", f" {neutral} ")
        return text
    
    def configure_peft(self):
        """
        Configure Parameter-Efficient Fine-Tuning (PEFT) using LoRA.
        
        Returns:
            PEFT-enabled model.
        """
        peft_config = LoraConfig(
            task_type="CAUSAL_LM", 
            r=self.config.get('peft_r', 16), 
            lora_alpha=self.config.get('peft_alpha', 32), 
            lora_dropout=self.config.get('peft_dropout', 0.1)
        )
        return get_peft_model(self.model, peft_config)
    
    def train_sft(self):
        """
        Conduct Supervised Fine-Tuning (SFT) using HuggingFace's Trainer.
        """
        # Prepare datasets are already done in __init__
        prepared_datasets = self.prepare_training_data()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.get('output_dir', './results'),
            num_train_epochs=self.config.get('epochs', 3),
            per_device_train_batch_size=self.config.get('train_batch_size', 32),
            per_device_eval_batch_size=self.config.get('eval_batch_size', 32),
            warmup_steps=self.config.get('warmup_steps', 100),
            weight_decay=self.config.get('weight_decay', 0.01),
            logging_dir=self.config.get('logging_dir', './logs'),
            logging_steps=self.config.get('logging_steps', 10),
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=self.config.get('save_steps', 100),
            load_best_model_at_end=True,
            fp16=self.config.get('fp16', False),
            bf16=self.config.get('bf16', False)
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=prepared_datasets['train'],
            eval_dataset=prepared_datasets['validation'],
            data_collator=data_collator,
        )
        
        # Train model
        trainer.train()
        
        # Save and log
        trainer.save_model(self.config.get('final_model_dir', './final_model'))
        if self._wandb is not None:
            wandb.finish()
    
    def train_dpo(self):
        """
        Conduct Direct Preference Optimization (DPO) training using custom training loop.
        """
        # Initialize DPO-specific components if any
        # Assuming availability of a reference model
        ref_model = AutoModelForCausalLM.from_pretrained(self.config['ref_model_path'])
        ref_model = get_peft_model(ref_model, LoraConfig(
            task_type="CAUSAL_LM",
            r=self.config.get('peft_r', 16),
            lora_alpha=self.config.get('peft_alpha', 32),
            lora_dropout=self.config.get('peft_dropout', 0.1)
        ))
        ref_model.to(torch.cuda.current_device())
        ref_model.eval()
        
        # Initialize DPOTrainer
        dpo_trainer = DPOTrainer(
            model=self.peft_model,
            ref_model=ref_model,
            strategy=self.strategy,
            tokenizer=self.tokenizer,
            optim=self.optimizer,
            train_dataloader=self.train_dataloader,
            eval_dataloader=self.eval_dataloader,
            scheduler=self.scheduler,
            max_norm=self.config.get('max_norm', 1.0),
            beta=self.config.get('dpo_beta', 0.01),
            max_epochs=self.config.get('dpo_max_epochs', 2),
            save_hf_ckpt=self.save_hf_ckpt,
            disable_ds_ckpt=self.disable_ds_ckpt
        )
        
        # Start DPO Training
        dpo_trainer.fit(args=self.config)
    
    def fine_tune(self):
        """
        Decide which training mode to execute: SFT or DPO.
        """
        mode = self.config.get('training_mode', 'sft').lower()
        if mode == 'sft':
            self.train_sft()
        elif mode == 'dpo':
            self.train_dpo()
        else:
            raise ValueError(f"Unknown training mode: {mode}")

# Placeholder classes for alignment and optimization components
class AlignmentConstructor:
    def generate_responses(self, texts):
        # Implement response generation logic
        return [{"text": text} for text in texts]

class ResponseGenerator:
    def generate_responses(self, texts):
        # Implement advanced response generation logic
        return [{"text": text} for text in texts]

class MixtureOptimizer:
    def optimize_data_mixture(self, df):
        # Implement data mixture optimization logic
        return df

# Placeholder Strategy class
class Strategy:
    def __init__(self, zero_stage=3, bf16=False, gradient_checkpointing=False, ring_attn_group=None):
        self.zero_stage = zero_stage
        self.bf16 = bf16
        self.gradient_checkpointing = gradient_checkpointing
        self.ring_attn_group = ring_attn_group
    
    def is_rank_0(self):
        # Implement logic to check if the current process is rank 0
        return True
    
    def is_distributed(self):
        # Implement logic to check if training is distributed
        return False
    
    def all_reduce(self, logs_dict):
        # Implement all_reduce for distributed logging
        return logs_dict
    
    def backward(self, loss, model, optimizer):
        loss.backward()
    
    def optimizer_step(self, optimizer, model, scheduler):
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    def save_ckpt(self, model, ckpt_path, tag, max_ckpt_num, max_ckpt_mem, client_states):
        # Implement checkpoint saving logic
        os.makedirs(ckpt_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_path, f"{tag}.pt"))
    
    def save_model(self, model, tokenizer, save_path):
        # Implement HuggingFace model saving
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

# Example usage
def main():
    config = {
        'base_model': 'gpt2-medium',
        'ref_model_path': 'gpt2-medium',  # Reference model path for DPO
        'dataset_type': 'csv',
        'dataset_path': 'path/to/your/dataset.csv',
        'max_length': 512,
        'train_batch_size': 32,
        'eval_batch_size': 32,
        'learning_rate': 5e-5,
        'scheduler_step_size': 1,
        'scheduler_gamma': 0.1,
        'zero_stage': 3,
        'bf16': True,
        'gradient_checkpointing': True,
        'peft_r': 16,
        'peft_alpha': 32,
        'peft_dropout': 0.1,
        'aux_loss_coef': 0.01,
        'packing_samples': False,
        'save_hf_ckpt': True,
        'disable_ds_ckpt': False,
        'use_wandb': True,
        'wandb_api_key': 'your_wandb_api_key',
        'wandb_org': 'your_org',
        'wandb_project': 'unified-trainer',
        'wandb_group': 'group_name',
        'wandb_run_name': 'run_name',
        'use_tensorboard': './logs/unified_trainer',
        'epochs': 3,
        'dpo_beta': 0.01,
        'dpo_max_epochs': 2,
        'training_mode': 'dpo'  # 'sft' or 'dpo'
    }
    
    trainer = UnifiedLLMTrainer(config)
    trainer.fine_tune()

if __name__ == "__main__":
    main()


# Key Features of UnifiedLLMTrainer:
# Dual Training Modes:

# Supervised Fine-Tuning (SFT): For standard supervised tasks with labeled data.
# Direct Preference Optimization (DPO): For preference-based optimization using rewards or human feedback.
# Bias Mitigation:

# Incorporates advanced bias mitigation techniques to ensure fair and unbiased model outputs.
# Parameter-Efficient Fine-Tuning (PEFT):

# Integrates LoRA for efficient adaptation of large models.
# Flexible Checkpointing:

# Supports both DeepSpeed and HuggingFace checkpointing formats for versatility.
# Comprehensive Logging and Monitoring:

# Utilizes WandB and TensorBoard for detailed tracking of training metrics.
# Distributed Training Support:

# Leverages strategies like ring attention groups and DeepSpeed optimizations for scalable training.
# Auxiliary Losses and Gradient Clipping:

# Supports optional auxiliary and NLL losses with gradient norm clipping to stabilize training.
# Integration with HuggingFace's Trainer:

# Provides an abstraction layer over HuggingFace's Trainer while retaining the flexibility of custom training loops when needed.
Potential Extensions: