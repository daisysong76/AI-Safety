import pandas as pd
import numpy as np
from transformers import pipeline

class GenderDebiasDatasetConstructor:
    def __init__(self):
        # Bias detection pipeline
        self.bias_detector = pipeline('text-classification', 
                                      model='unitary/toxic-bert')
    
    def debias_text(self, text: str) -> str:
        """
        Apply gender debiasing techniques to text
        
        Args:
        - text: Input text
        
        Returns:
        Debiased text
        """
        # Replace gendered pronouns
        pronouns = {
            'he': 'they',
            'she': 'they',
            'him': 'them',
            'her': 'them',
            'his': 'their',
            'hers': 'theirs'
        }
        
        for gendered, neutral in pronouns.items():
            text = text.replace(gendered, neutral)
        
        return text
    
    def detect_gender_bias(self, text: str) -> float:
        """
        Quantify gender bias in text
        
        Args:
        - text: Input text
        
        Returns:
        Bias score (0-1)
        """
        results = self.bias_detector(text)
        return results[0]['score']
    
    def construct_debiased_dataset(self, 
                                   original_dataset: pd.DataFrame,
                                   bias_threshold: float = 0.3) -> pd.DataFrame:
        """
        Create gender-debiased dataset
        
        Args:
        - original_dataset: Input dataset
        - bias_threshold: Maximum acceptable bias score
        
        Returns:
        Debiased dataset
        """
        # Apply debiasing
        debiased_data = []
        
        for _, row in original_dataset.iterrows():
            debiased_text = self.debias_text(row['text'])
            bias_score = self.detect_gender_bias(debiased_text)
            
            if bias_score < bias_threshold:
                row['text'] = debiased_text
                row['bias_score'] = bias_score
                debiased_data.append(row)
        
        return pd.DataFrame(debiased_data)

def main():
    # Example usage
    original_data = pd.DataFrame({
        'text': [
            "The businessman worked hard in his office.",
            "She is an excellent engineer at the tech company.",
            "He made a breakthrough in medical research."
        ]
    })
    
    debiaser = GenderDebiasDatasetConstructor()
    debiased_dataset = debiaser.construct_debiased_dataset(original_data)
    
    print(debiased_dataset)

if __name__ == "__main__":
    main()