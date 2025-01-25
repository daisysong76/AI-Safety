# pip install wandb sklearn typing_extensions
# wandb login

import os
import pandas as pd
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from datasets import load_dataset, DatasetDict

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    Trainer, 
    pipeline,
    EarlyStoppingCallback,
    AutoConfig
)

from accelerate import Accelerator
from peft import (
    LoraConfig, 
    get_peft_model,
    PeftConfig,
    PeftModel,
    prepare_model_for_kbit_training
)
from transformers import DataCollatorForSeq2Seq
from sklearn.metrics import classification_report, confusion_matrix
import wandb
from torch.utils.data import WeightedRandomSampler
import logging
import json
from dataclasses import dataclass
from torch.nn import functional as F

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class ModelConfig:
    """Configuration class for model training parameters"""
    model_name: str = "meta-llama/Llama-2-13b-chat-hf"
    dataset_path: str = "data/processed/synthetic_content_safety_data.csv"
    output_dir: str = "llama-content-safety"
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 5e-5
    max_seq_length: int = 256
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    mixed_precision: str = 'fp16'
    quantization: str = '8bit'  # Options: '8bit', '4bit', None

class ContentSafetyModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.accelerator = Accelerator(
            mixed_precision=config.mixed_precision,
            gradient_accumulation_steps=config.gradient_accumulation_steps
        )
        self.device = self.accelerator.device
        self.setup_wandb()
        
    def setup_wandb(self):
        """Initialize Weights & Biases tracking"""
        wandb.init(
            project="content-safety-classification",
            config=self.config.__dict__
        )

    def load_and_prepare_data(self) -> Tuple[DatasetDict, DatasetDict, DatasetDict]:
        """Load and prepare dataset with advanced preprocessing"""
        logging.info("Loading dataset...")
        data = pd.read_csv(self.config.dataset_path)
        
        # Advanced data validation and cleaning
        self._validate_data(data)
        data = self._clean_data(data)
        
        # Handle class imbalance
        class_weights = self._calculate_class_weights(data['label'])
        
        # Stratified split
        train_data, remaining = self._stratified_split(data, test_size=0.3)
        val_data, test_data = self._stratified_split(remaining, test_size=0.5)
        
        # Convert to DatasetDict format with weights
        datasets = {
            'train': self._create_weighted_dataset(train_data, class_weights),
            'validation': DatasetDict.from_pandas(val_data),
            'test': DatasetDict.from_pandas(test_data)
        }
        
        self._log_data_stats(datasets)
        return datasets['train'], datasets['validation'], datasets['test']

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate dataset structure and contents"""
        required_columns = ['text', 'label']
        assert all(col in data.columns for col in required_columns), \
            f"Missing required columns. Expected: {required_columns}"
        
        # Validate data types and ranges
        assert data['text'].dtype == object, "Text column must be string type"
        assert data['label'].dtype in [int, float], "Label column must be numeric"
        
        # Check for missing values
        missing_values = data.isnull().sum()
        if missing_values.any():
            logging.warning(f"Found missing values:\n{missing_values[missing_values > 0]}")

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply advanced data cleaning techniques"""
        # Remove duplicates
        data = data.drop_duplicates(subset=['text'])
        
        # Clean text
        data['text'] = data['text'].apply(self._clean_text)
        
        # Remove extreme outliers in text length
        text_lengths = data['text'].str.len()
        q1, q3 = text_lengths.quantile([0.25, 0.75])
        iqr = q3 - q1
        data = data[
            (text_lengths >= q1 - 1.5 * iqr) & 
            (text_lengths <= q3 + 1.5 * iqr)
        ]
        
        return data

    @staticmethod
    def _clean_text(text: str) -> str:
        """Apply text cleaning operations"""
        import re
        # Remove special characters but keep essential punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        # Normalize whitespace
        text = ' '.join(text.split())
        return text

    def setup_model(self):
        """Initialize and configure the model with advanced settings"""
        logging.info("Setting up model and tokenizer...")
        
        # Load configuration
        model_config = AutoConfig.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # Initialize tokenizer with special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            padding_side="right",
            truncation_side="right",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with quantization if specified
        load_kwargs = {"device_map": "auto"}
        if self.config.quantization == '8bit':
            load_kwargs["load_in_8bit"] = True
        elif self.config.quantization == '4bit':
            load_kwargs["load_in_4bit"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            config=model_config,
            **load_kwargs
        )
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        
        # Enable gradient checkpointing
        self.model.gradient_checkpointing_enable()
        
        return self.model, self.tokenizer

    def train(self, train_dataset, val_dataset):
        """Train the model with advanced training configurations"""
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.epochs,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            fp16=self.config.mixed_precision == 'fp16',
            logging_dir=f"{self.config.output_dir}/logs",
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="wandb"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=self._get_data_collator(),
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        trainer.train()
        return trainer

    def _get_data_collator(self):
        """Create a custom data collator with dynamic padding"""
        return DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            max_length=self.config.max_seq_length
        )

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics"""
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=-1)
        
        # Calculate detailed metrics
        report = classification_report(
            labels, 
            predictions, 
            output_dict=True,
            zero_division=0
        )
        
        # Log confusion matrix to wandb
        cm = confusion_matrix(labels, predictions)
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=labels,
            preds=predictions
        )})
        
        # Return aggregated metrics
        metrics = {
            "accuracy": report['accuracy'],
            "macro_f1": report['macro avg']['f1-score'],
            "weighted_f1": report['weighted avg']['f1-score']
        }
        
        # Log detailed metrics
        wandb.log(metrics)
        return metrics

    def save_model(self, output_dir: str):
        """Save model with additional artifacts"""
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save configuration
        config_path = os.path.join(output_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        # Save training metrics history
        metrics_path = os.path.join(output_dir, 'training_metrics.json')
        wandb.save(metrics_path)

    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """Make predictions with confidence scores"""
        predict_pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        results = []
        for text in texts:
            # Get raw logits
            outputs = predict_pipeline(text, return_all_scores=True)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(torch.tensor(outputs[0]), dim=0)
            
            # Create prediction object with confidence
            prediction = {
                'label': int(torch.argmax(probabilities)),
                'confidence': float(torch.max(probabilities)),
                'probabilities': {
                    f'class_{i}': float(p) 
                    for i, p in enumerate(probabilities)
                }
            }
            results.append(prediction)
            
        return results

def main():
    # Initialize configuration
    config = ModelConfig()
    
    # Initialize model
    content_safety = ContentSafetyModel(config)
    
    # Load and prepare data
    train_dataset, val_dataset, test_dataset = content_safety.load_and_prepare_data()
    
    # Setup model and tokenizer
    model, tokenizer = content_safety.setup_model()
    
    # Train model
    trainer = content_safety.train(train_dataset, val_dataset)
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    logging.info(f"Test Results: {test_results}")
    
    # Save model and artifacts
    content_safety.save_model(config.output_dir)
    
    # Example prediction
    sample_texts = [
        "This is a toxic statement.",
        "This is a positive statement."
    ]
    predictions = content_safety.predict(sample_texts)
    logging.info(f"Predictions: {predictions}")
    
    # Clean up
    wandb.finish()

if __name__ == "__main__":
    main()