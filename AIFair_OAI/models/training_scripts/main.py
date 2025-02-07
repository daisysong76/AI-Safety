import os
import sys
import logging
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import torch
import wandb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.logging import RichHandler
from sklearn.model_selection import train_test_split

# Import our modules
from content_safety_model import ContentSafetyModel, ModelConfig
from bias_module import BiasAnalyzer, BiasReducer, BiasConfig, CounterfactualGenerator
from data_generator import ContentSafetyDataGenerator

class ContentSafetySystem:
    """Advanced Content Safety System with comprehensive functionality"""
    
    def __init__(self, config_path: str, experiment_name: Optional[str] = None):
        """Initialize the system with configuration and setup"""
        self.console = Console()
        self._setup_logging()
        self.config = self._load_config(config_path)
        self.experiment_name = experiment_name or f"experiment_{int(time.time())}"
        self.device = self._setup_device()
        self._initialize_components()
        self._setup_monitoring()

    def _setup_logging(self):
        """Setup enhanced logging with Rich formatting"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[
                RichHandler(rich_tracebacks=True),
                logging.FileHandler("system.log")
            ]
        )
        self.logger = logging.getLogger("content_safety")

    def _load_config(self, config_path: str) -> Dict:
        """Load and validate configuration"""
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            required_sections = ['model', 'bias', 'data', 'training']
            for section in required_sections:
                if section not in config:
                    raise ValueError(f"Missing required config section: {section}")
            
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config: {str(e)}")
            raise

    def _setup_device(self) -> torch.device:
        """Setup and optimize GPU/CPU device"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # Set optimal CUDA settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            n_gpu = torch.cuda.device_count()
            self.logger.info(f"Using {n_gpu} GPU(s)")
        else:
            device = torch.device("cpu")
            self.logger.info("Using CPU")
        return device

    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Initialize model
            self.model_config = ModelConfig(**self.config['model'])
            self.model = ContentSafetyModel(self.model_config)
            
            # Initialize bias components
            self.bias_config = BiasConfig(**self.config['bias'])
            self.bias_analyzer = BiasAnalyzer(self.bias_config)
            self.bias_reducer = BiasReducer(self.bias_config)
            
            # Initialize data generator
            self.data_generator = ContentSafetyDataGenerator(
                seed=self.config['data'].get('seed', 42)
            )
            
            # Initialize counterfactual generator
            self.counterfactual_generator = CounterfactualGenerator(
                self.bias_config.protected_attributes
            )
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise

    def _setup_monitoring(self):
        """Setup monitoring and tracking"""
        wandb.init(
            project="content-safety-advanced",
            name=self.experiment_name,
            config=self.config,
            resume=True
        )
        
        # Setup system monitoring
        self.metrics = {
            'training_loss': [],
            'validation_metrics': [],
            'bias_metrics': [],
            'system_metrics': []
        }

    def prepare_data(self, 
                    data_path: Optional[str] = None,
                    use_synthetic: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Prepare and validate dataset"""
        with self.console.status("[bold green]Preparing data...") as status:
            try:
                if use_synthetic:
                    self.logger.info("Generating synthetic dataset")
                    data = self.data_generator.generate_dataset(
                        num_samples=self.config['data']['num_samples'],
                        class_distribution=self.config['data']['class_distribution']
                    )
                else:
                    self.logger.info(f"Loading dataset from {data_path}")
                    data = pd.read_csv(data_path)
                
                # Validate data
                self._validate_data(data)
                
                # Apply bias detection and mitigation
                fairness_metrics = self.bias_analyzer.compute_fairness_metrics(data)
                wandb.log({"initial_fairness_metrics": fairness_metrics})
                
                if fairness_metrics["disparate_impact"] < self.bias_config.fairness_threshold:
                    self.logger.info("Applying bias mitigation...")
                    data = self.bias_reducer.mitigate_bias(data)
                
                # Split data
                train_data, temp_data = train_test_split(
                    data, 
                    test_size=0.3, 
                    stratify=data['label']
                )
                val_data, test_data = train_test_split(
                    temp_data, 
                    test_size=0.5, 
                    stratify=temp_data['label']
                )
                
                self._log_data_stats(train_data, val_data, test_data)
                
                return train_data, val_data, test_data
                
            except Exception as e:
                self.logger.error(f"Data preparation failed: {str(e)}")
                raise

    def _validate_data(self, data: pd.DataFrame):
        """Validate dataset structure and content"""
        required_columns = ['text', 'label'] + self.bias_config.protected_attributes
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for data quality issues
        null_counts = data.isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Found null values:\n{null_counts[null_counts > 0]}")
            
        # Validate label distribution
        label_dist = data['label'].value_counts(normalize=True)
        self.logger.info(f"Label distribution:\n{label_dist}")

    def train(self, 
             train_data: pd.DataFrame, 
             val_data: pd.DataFrame,
             output_dir: str):
        """Train the model with comprehensive monitoring"""
        try:
            self.logger.info("Starting training process")
            
            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Setup training progress
            with Progress(
                SpinnerColumn(),
                *Progress.get_default_columns(),
                TimeElapsedColumn(),
                console=self.console
            ) as progress:
                
                # Generate counterfactuals for training
                task_id = progress.add_task("Generating counterfactuals...", total=None)
                augmented_train_data = self._augment_with_counterfactuals(train_data)
                progress.update(task_id, completed=True)
                
                # Train model
                task_id = progress.add_task("Training model...", total=self.config['training']['epochs'])
                
                def progress_callback(epoch, metrics):
                    progress.update(task_id, advance=1)
                    self._update_metrics(metrics)
                    wandb.log(metrics)
                
                trainer = self.model.train(
                    augmented_train_data,
                    val_data,
                    callbacks=[progress_callback]
                )
                
                # Evaluate final model
                task_id = progress.add_task("Evaluating model...", total=None)
                evaluation_results = self._evaluate_model(trainer, val_data)
                progress.update(task_id, completed=True)
                
                # Save model and artifacts
                task_id = progress.add_task("Saving model...", total=None)
                self._save_artifacts(output_dir, trainer, evaluation_results)
                progress.update(task_id, completed=True)
            
            return trainer, evaluation_results
            
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def _augment_with_counterfactuals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate counterfactual examples for training"""
        augmented_data = []
        
        for _, row in data.iterrows():
            augmented_data.append(row)
            
            # Generate counterfactuals for non-safe content
            if row['label'] != 0:
                counterfactuals = self.counterfactual_generator.generate_counterfactuals(
                    row['text'],
                    self.config['data']['attribute_values']
                )
                
                for cf_text in counterfactuals:
                    new_row = row.copy()
                    new_row['text'] = cf_text
                    augmented_data.append(new_row)
        
        return pd.DataFrame(augmented_data)

    def _evaluate_model(self, trainer, eval_data: pd.DataFrame) -> Dict:
        """Comprehensive model evaluation"""
        # Get predictions
        predictions = trainer.predict(eval_data)
        
        # Calculate metrics
        metrics = trainer.evaluate(eval_data)
        
        # Analyze predictions for bias
        bias_metrics = self.bias_analyzer.detect_bias_in_outputs(
            predictions.predictions,
            eval_data[self.bias_config.protected_attributes].values
        )
        
        # Combine all metrics
        evaluation_results = {
            "model_metrics": metrics,
            "bias_metrics": bias_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        wandb.log({"evaluation": evaluation_results})
        return evaluation_results

    def _save_artifacts(self, 
                       output_dir: Path, 
                       trainer, 
                       evaluation_results: Dict):
        """Save model artifacts and results"""
        # Save model
        model_dir = output_dir / "model"
        trainer.save_model(str(model_dir))
        
        # Save configurations
        config_path = output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save evaluation results
        eval_path = output_dir / "evaluation_results.json"
        with open(eval_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Save metrics history
        metrics_path = output_dir / "metrics_history.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Log artifacts to wandb
        wandb.save(str(config_path))
        wandb.save(str(eval_path))
        wandb.save(str(metrics_path))

    def predict(self, texts: List[str]) -> List[Dict]:
        """Make predictions with bias analysis"""
        try:
            # Get model predictions
            predictions = self.model.predict(texts)
            
            # Analyze predictions for bias
            bias_scores = self.bias_analyzer.detect_bias_in_outputs(
                predictions,
                self.bias_config.protected_attributes
            )
            
            # Combine results
            results = []
            for pred, text in zip(predictions, texts):
                result = {
                    "text": text,
                    "prediction": pred,
                    "bias_scores": bias_scores,
                    "warnings": self._generate_warnings(bias_scores)
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {str(e)}")
            raise

    def _generate_warnings(self, bias_scores: Dict[str, float]) -> List[str]:
        """Generate warning messages for high bias scores"""
        warnings = []
        for attr, score in bias_scores.items():
            if abs(score) > self.bias_config.fairness_threshold:
                warnings.append(f"High bias detected for {attr}: {score:.2f}")
        return warnings

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Content Safety System")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--data", help="Path to input data")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--experiment", help="Experiment name")
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = ContentSafetySystem(args.config, args.experiment)
        
        # Prepare data
        train_data, val_data, test_data = system.prepare_data(
            data_path=args.data,
            use_synthetic=args.synthetic
        )
        
        # Train model
        trainer, results = system.train(train_data, val_data, args.output)
        
        # Final evaluation
        system.logger.info("Final evaluation results:")
        system.logger.info(json.dumps(results, indent=2))
        
        # Example prediction
        sample_texts = [
            "This is a test message",
            "Another test message"
        ]
        predictions = system.predict(sample_texts)
        system.logger.info("Sample predictions:")
        system.logger.info(json.dumps(predictions, indent=2))
        
        # Cleanup
        wandb.finish()
        
    except Exception as e:
        logging.error(f"System failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()