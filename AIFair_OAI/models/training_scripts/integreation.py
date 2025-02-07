# main.py
import os
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import logging
import wandb
from bias_module import BiasAnalyzer, BiasReducer, BiasConfig, CounterfactualGenerator
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments
)
from content_safety_model import ContentSafetyModel, ModelConfig
from data_generator import ContentSafetyDataGenerator
import json

@dataclass
class IntegratedConfig:
    """Combined configuration for the entire system"""
    model_config: ModelConfig
    bias_config: BiasConfig
    data_config: Dict[str, any]
    
    @classmethod
    def from_json(cls, config_path: str) -> "IntegratedConfig":
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(
            model_config=ModelConfig(**config_dict['model']),
            bias_config=BiasConfig(**config_dict['bias']),
            data_config=config_dict['data']
        )

class IntegratedContentSafetySystem:
    """Main class integrating all components of the content safety system"""
    
    def __init__(self, config: IntegratedConfig):
        self.config = config
        self._setup_logging()
        self._initialize_components()
        self._setup_wandb()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('integrated_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_components(self):
        """Initialize all system components"""
        # Initialize content safety model
        self.content_model = ContentSafetyModel(self.config.model_config)
        
        # Initialize bias detection and mitigation components
        self.bias_analyzer = BiasAnalyzer(self.config.bias_config)
        self.bias_reducer = BiasReducer(self.config.bias_config)
        
        # Initialize counterfactual generator
        self.counterfactual_generator = CounterfactualGenerator(
            self.config.bias_config.protected_attributes
        )
        
        # Initialize data generator if needed
        self.data_generator = ContentSafetyDataGenerator(
            seed=self.config.data_config.get('seed', 42)
        )
    
    def _setup_wandb(self):
        """Initialize Weights & Biases tracking"""
        wandb.init(
            project="integrated-content-safety",
            config={
                "model": self.config.model_config.__dict__,
                "bias": self.config.bias_config.__dict__,
                "data": self.config.data_config
            }
        )
    
    def prepare_data(self, 
                    data_path: Optional[str] = None, 
                    use_synthetic: bool = False) -> pd.DataFrame:
        """Prepare dataset with bias awareness"""
        if use_synthetic:
            self.logger.info("Generating synthetic dataset...")
            df = self.data_generator.generate_dataset(
                num_samples=self.config.data_config['num_samples'],
                class_distribution=self.config.data_config['class_distribution']
            )
        else:
            self.logger.info(f"Loading dataset from {data_path}")
            df = pd.read_csv(data_path)
        
        # Prepare dataset for bias analysis
        dataset = self.bias_analyzer.prepare_dataset(df, "label")
        
        # Check for bias and apply mitigation if needed
        fairness_metrics = self.bias_analyzer.compute_fairness_metrics(dataset)
        self.logger.info(f"Initial fairness metrics: {fairness_metrics}")
        
        if fairness_metrics["disparate_impact"] < self.config.bias_config.fairness_threshold:
            self.logger.info("Applying bias mitigation...")
            dataset = self.bias_reducer.mitigate_bias(dataset)
            new_metrics = self.bias_analyzer.compute_fairness_metrics(dataset)
            self.logger.info(f"Post-mitigation metrics: {new_metrics}")
        
        return dataset
    
    def train(self, train_dataset, val_dataset):
        """Train the model with bias awareness"""
        self.logger.info("Starting training process...")
        
        # Generate counterfactuals for sensitive examples
        augmented_dataset = self._augment_with_counterfactuals(train_dataset)
        
        # Train the model
        trainer = self.content_model.train(augmented_dataset, val_dataset)
        
        # Analyze model outputs for bias
        self._analyze_training_outputs(trainer)
        
        return trainer
    
    def _augment_with_counterfactuals(self, dataset) -> pd.DataFrame:
        """Augment training data with counterfactuals"""
        self.logger.info("Generating counterfactual examples...")
        
        augmented_data = []
        attribute_values = self.config.data_config.get('attribute_values', {})
        
        for idx, row in dataset.iterrows():
            augmented_data.append(row)
            
            # Generate counterfactuals for sensitive examples
            if row['label'] != 0:  # Non-safe content
                counterfactuals = self.counterfactual_generator.generate_counterfactuals(
                    row['text'],
                    attribute_values
                )
                
                for cf_text in counterfactuals:
                    new_row = row.copy()
                    new_row['text'] = cf_text
                    augmented_data.append(new_row)
        
        return pd.DataFrame(augmented_data)
    
    def _analyze_training_outputs(self, trainer):
        """Analyze model outputs for bias during training"""
        eval_results = trainer.evaluate()
        
        # Analyze predictions for bias
        predictions = trainer.predict(trainer.eval_dataset)
        bias_metrics = self.bias_analyzer.detect_bias_in_outputs(
            predictions.predictions,
            self.config.bias_config.protected_attributes
        )
        
        wandb.log({
            "eval_results": eval_results,
            "bias_metrics": bias_metrics
        })
    
    def predict(self, texts: List[str]) -> List[Dict]:
        """Make predictions with bias checking"""
        # Get base predictions
        predictions = self.content_model.predict(texts)
        
        # Analyze predictions for bias
        bias_scores = self.bias_analyzer.detect_bias_in_outputs(
            predictions,
            self.config.bias_config.protected_attributes
        )
        
        # Combine predictions with bias information
        results = []
        for pred, text in zip(predictions, texts):
            result = {
                "prediction": pred,
                "bias_scores": bias_scores,
                "warnings": self._generate_bias_warnings(bias_scores)
            }
            results.append(result)
        
        return results
    
    def _generate_bias_warnings(self, bias_scores: Dict[str, float]) -> List[str]:
        """Generate warning messages for high bias scores"""
        warnings = []
        threshold = self.config.bias_config.fairness_threshold
        
        for term, score in bias_scores.items():
            if abs(score) > threshold:
                warnings.append(
                    f"High bias detected for term '{term}' with score {score:.2f}"
                )
        
        return warnings
    
    def save(self, output_dir: str):
        """Save all components and configurations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        self.content_model.save_model(os.path.join(output_dir, "model"))
        
        # Save configurations
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump({
                "model": self.config.model_config.__dict__,
                "bias": self.config.bias_config.__dict__,
                "data": self.config.data_config
            }, f, indent=2)
        
        # Save bias analysis results
        bias_path = os.path.join(output_dir, "bias_analysis.json")
        with open(bias_path, 'w') as f:
            json.dump({
                "fairness_metrics": self.bias_analyzer.metrics_history,
                "mitigation_history": self.bias_reducer.mitigation_history
            }, f, indent=2)
        
        self.logger.info(f"System saved to {output_dir}")

def main():
    # Load configuration
    config = IntegratedConfig.from_json("config.json")
    
    # Initialize integrated system
    system = IntegratedContentSafetySystem(config)
    
    # Prepare data
    train_dataset = system.prepare_data(
        data_path="path/to/train.csv",
        use_synthetic=False
    )
    val_dataset = system.prepare_data(
        data_path="path/to/val.csv",
        use_synthetic=False
    )
    
    # Train model
    trainer = system.train(train_dataset, val_dataset)
    
    # Save system
    system.save("output_directory")
    
    # Example prediction
    texts = [
        "This is a test message",
        "Another test message"
    ]
    predictions = system.predict(texts)
    print("Predictions:", predictions)

if __name__ == "__main__":
    main()