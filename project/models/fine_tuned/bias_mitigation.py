import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import confusion_matrix
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.algorithms.inprocessing import AdversarialDebiasing
import tensorflow as tf
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
import torch
from torch import nn
import logging
from dataclasses import dataclass
import json
from collections import defaultdict

@dataclass
class BiasConfig:
    """Configuration for bias detection and mitigation"""
    protected_attributes: List[str]
    favorable_label: int = 1
    unfavorable_label: int = 0
    privileged_groups: List[Dict]
    unprivileged_groups: List[Dict]
    mitigation_method: str = "reweighing"  # Options: reweighing, adversarial, postprocessing
    fairness_threshold: float = 0.1
    debiasing_strength: float = 0.5

class BiasAnalyzer:
    """Class for detecting and analyzing bias in model outputs"""
    
    def __init__(self, config: BiasConfig):
        self.config = config
        self.metrics_history = defaultdict(list)
        self._setup_logging()
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('bias_analysis.log'),
                logging.StreamHandler()
            ]
        )
    
    def prepare_dataset(self, 
                       df: pd.DataFrame, 
                       label_column: str) -> BinaryLabelDataset:
        """Convert pandas DataFrame to AIF360 BinaryLabelDataset"""
        return BinaryLabelDataset(
            df=df,
            label_names=[label_column],
            protected_attribute_names=self.config.protected_attributes,
            favorable_label=self.config.favorable_label,
            unfavorable_label=self.config.unfavorable_label
        )
    
    def compute_fairness_metrics(self, 
                               dataset: BinaryLabelDataset) -> Dict[str, float]:
        """Compute comprehensive fairness metrics"""
        metric = BinaryLabelDatasetMetric(
            dataset,
            unprivileged_groups=self.config.unprivileged_groups,
            privileged_groups=self.config.privileged_groups
        )
        
        metrics = {
            "disparate_impact": metric.disparate_impact(),
            "statistical_parity_difference": metric.statistical_parity_difference(),
            "equal_opportunity_difference": metric.equal_opportunity_difference(),
            "average_odds_difference": metric.average_odds_difference(),
            "theil_index": metric.theil_index()
        }
        
        self.metrics_history["fairness_metrics"].append(metrics)
        return metrics
    
    def detect_bias_in_outputs(self, 
                             outputs: List[str], 
                             sensitive_terms: List[str]) -> Dict[str, float]:
        """Analyze model outputs for potential biases"""
        bias_scores = defaultdict(float)
        
        for output in outputs:
            # Analyze sentiment and context around sensitive terms
            for term in sensitive_terms:
                if term.lower() in output.lower():
                    # Implement sentiment analysis around the term
                    context = self._extract_context(output, term)
                    sentiment_score = self._analyze_sentiment(context)
                    bias_scores[term] += sentiment_score
        
        # Normalize scores
        for term in bias_scores:
            bias_scores[term] /= len(outputs)
        
        return dict(bias_scores)
    
    def _extract_context(self, text: str, term: str, window: int = 5) -> str:
        """Extract context window around sensitive term"""
        words = text.split()
        try:
            idx = [i for i, word in enumerate(words) if term.lower() in word.lower()][0]
            start = max(0, idx - window)
            end = min(len(words), idx + window + 1)
            return " ".join(words[start:end])
        except IndexError:
            return ""
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text context"""
        # Implement sentiment analysis (placeholder)
        # Could use transformers pipeline or other sentiment analyzer
        return 0.0

class BiasReducer:
    """Class for implementing bias mitigation techniques"""
    
    def __init__(self, config: BiasConfig):
        self.config = config
        self.mitigation_history = defaultdict(list)
    
    def mitigate_bias(self, 
                      dataset: BinaryLabelDataset, 
                      method: Optional[str] = None) -> BinaryLabelDataset:
        """Apply specified bias mitigation technique"""
        method = method or self.config.mitigation_method
        
        if method == "reweighing":
            return self._apply_reweighing(dataset)
        elif method == "adversarial":
            return self._apply_adversarial_debiasing(dataset)
        elif method == "postprocessing":
            return self._apply_postprocessing(dataset)
        else:
            raise ValueError(f"Unknown mitigation method: {method}")
    
    def _apply_reweighing(self, dataset: BinaryLabelDataset) -> BinaryLabelDataset:
        """Apply reweighing technique"""
        reweighing = Reweighing(
            unprivileged_groups=self.config.unprivileged_groups,
            privileged_groups=self.config.privileged_groups
        )
        return reweighing.fit_transform(dataset)
    
    def _apply_adversarial_debiasing(self, 
                                   dataset: BinaryLabelDataset) -> BinaryLabelDataset:
        """Apply adversarial debiasing"""
        sess = tf.Session()
        debiaser = AdversarialDebiasing(
            privileged_groups=self.config.privileged_groups,
            unprivileged_groups=self.config.unprivileged_groups,
            scope_name='debiased_classifier',
            debias_strength=self.config.debiasing_strength,
            sess=sess
        )
        return debiaser.fit_transform(dataset)
    
    def _apply_postprocessing(self, dataset: BinaryLabelDataset) -> BinaryLabelDataset:
        """Apply post-processing bias mitigation"""
        # Implement calibrated equalized odds post-processing
        return dataset

class CounterfactualGenerator:
    """Class for generating counterfactual examples"""
    
    def __init__(self, protected_attributes: List[str]):
        self.protected_attributes = protected_attributes
    
    def generate_counterfactuals(self, 
                               text: str, 
                               attribute_values: Dict[str, List[str]]) -> List[str]:
        """Generate counterfactual examples by varying protected attributes"""
        counterfactuals = []
        
        for attr in self.protected_attributes:
            if attr in attribute_values:
                for value in attribute_values[attr]:
                    # Generate counterfactual by replacing attribute mentions
                    new_text = self._replace_attribute(text, attr, value)
                    counterfactuals.append(new_text)
        
        return counterfactuals
    
    def _replace_attribute(self, text: str, attribute: str, new_value: str) -> str:
        """Replace attribute mentions in text with new value"""
        # Implement attribute replacement logic
        return text

class ContrastiveLearningModule(nn.Module):
    """Module for implementing contrastive learning"""
    
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def contrastive_loss(self, 
                        anchor: torch.Tensor, 
                        positive: torch.Tensor, 
                        negative: torch.Tensor, 
                        margin: float = 1.0) -> torch.Tensor:
        """Compute triplet loss"""
        pos_dist = torch.norm(anchor - positive, dim=1)
        neg_dist = torch.norm(anchor - negative, dim=1)
        loss = torch.mean(torch.relu(pos_dist - neg_dist + margin))
        return loss

def main():
    # Example usage
    config = BiasConfig(
        protected_attributes=["gender", "race"],
        privileged_groups=[{"gender": 1}, {"race": 1}],
        unprivileged_groups=[{"gender": 0}, {"race": 0}],
        mitigation_method="reweighing"
    )
    
    # Initialize analyzers
    bias_analyzer = BiasAnalyzer(config)
    bias_reducer = BiasReducer(config)
    
    # Load sample data (replace with actual data)
    df = pd.DataFrame({
        "text": ["sample text 1", "sample text 2"],
        "label": [1, 0],
        "gender": [1, 0],
        "race": [1, 0]
    })
    
    # Prepare dataset
    dataset = bias_analyzer.prepare_dataset(df, "label")
    
    # Analyze bias
    fairness_metrics = bias_analyzer.compute_fairness_metrics(dataset)
    logging.info(f"Initial fairness metrics: {fairness_metrics}")
    
    # Apply bias mitigation
    debiased_dataset = bias_reducer.mitigate_bias(dataset)
    
    # Analyze results
    new_metrics = bias_analyzer.compute_fairness_metrics(debiased_dataset)
    logging.info(f"Post-mitigation metrics: {new_metrics}")
    
    # Save results
    with open("bias_analysis_results.json", "w") as f:
        json.dump({
            "initial_metrics": fairness_metrics,
            "post_mitigation_metrics": new_metrics
        }, f, indent=2)

if __name__ == "__main__":
    main()