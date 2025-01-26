import pandas as pd
from config import Config

def preprocess_dataset(input_path, output_path):
    data = pd.read_csv(input_path)
    # Example: Remove duplicates and clean text
    data = data.drop_duplicates().dropna()
    data['text'] = data['text'].str.lower()
    data.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_dataset(Config.RAW_DATA_DIR + "raw_data.csv", 
                       Config.PROCESSED_DATA_DIR + "cleaned_data.csv")


#How to Use This Script
# Place your raw data file in the data/raw/ directory (e.g., raw_data.csv).

# Run the script from the project root directory:
# python data/scripts/preprocess_data.py
# The cleaned data will be saved to data/processed/cleaned_data.csv.

# Advantages of This Placement
# Modularity: Keeps preprocessing logic separate from other components.
# Reusability: The function can be easily imported and reused in other scripts (e.g., for pipeline automation).
# Scalability: You can add more preprocessing functions or scripts in the same directory for different datasets or tasks.

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import transformers
import torch

class AlignmentDataConstructor:
    def __init__(self, 
                 diversity_strategies: List[str] = ['topic', 'complexity', 'perspective'],
                 max_diversity_clusters: int = 10):
        """
        Advanced alignment data construction pipeline
        
        Args:
        - diversity_strategies: Dimensions to ensure data diversity
        - max_diversity_clusters: Maximum number of diversity clusters
        """
        self.diversity_strategies = diversity_strategies
        self.max_diversity_clusters = max_diversity_clusters
        
        # Semantic embedding model
        self.embedding_model = transformers.AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
    def collect_raw_data(self, 
                         sources: List[str], 
                         filters: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Multi-source data collection with advanced filtering
        
        Args:
        - sources: List of data sources (APIs, databases, etc.)
        - filters: Sophisticated filtering criteria
        
        Returns:
        Preprocessed and filtered DataFrame
        """
        collected_data = []
        
        for source in sources:
            # Placeholder for source-specific data collection
            source_data = self._collect_from_source(source, filters)
            collected_data.append(source_data)
        
        return pd.concat(collected_data, ignore_index=True)
    
    def classify_instructions(self, 
                              data: pd.DataFrame, 
                              classification_model: Any = None) -> pd.DataFrame:
        """
        Advanced instruction classification
        
        Args:
        - data: Input data DataFrame
        - classification_model: Custom classification model
        
        Returns:
        DataFrame with instruction classifications
        """
        if classification_model is None:
            # Default to zero-shot classification
            classifier = transformers.pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli"
            )
        
        def classify_instruction(text):
            # Example classification categories
            categories = [
                'reasoning', 'creative', 'analytical', 
                'task-specific', 'open-ended', 'constrained'
            ]
            
            results = classifier(text, categories)
            return results['labels'][0]
        
        data['instruction_type'] = data['instruction'].apply(classify_instruction)
        return data
    
    def ensure_diversity(self, 
                         data: pd.DataFrame, 
                         embedding_dim: int = 384) -> pd.DataFrame:
        """
        Sophisticated diversity control mechanism
        
        Args:
        - data: Input data
        - embedding_dim: Embedding vector dimension
        
        Returns:
        Diversity-balanced dataset
        """
        # Generate embeddings
        embeddings = self._generate_embeddings(data['instruction'])
        
        # Cluster-based diversity sampling
        kmeans = KMeans(n_clusters=min(self.max_diversity_clusters, len(data)))
        kmeans.fit(embeddings)
        data['cluster'] = kmeans.labels_
        
        # Sample strategy: Ensure representation from each cluster
        diverse_data = data.groupby('cluster').apply(
            lambda x: x.sample(min(len(x), 10))
        ).reset_index(drop=True)
        
        return diverse_data
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate semantic embeddings for texts
        
        Args:
        - texts: List of text instructions
        
        Returns:
        Embedding matrix
        """
        with torch.no_grad():
            embeddings = self.embedding_model(
                transformers.AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')(
                    texts, 
                    padding=True, 
                    truncation=True, 
                    return_tensors='pt'
                )
            ).last_hidden_state.mean(dim=1)
        
        return embeddings.numpy()
    
    def construct_alignment_dataset(self, 
                                    sources: List[str], 
                                    filters: Dict[str, Any] = None) -> pd.DataFrame:
        """
        End-to-end alignment data construction pipeline
        
        Args:
        - sources: Data sources
        - filters: Collection filters
        
        Returns:
        Processed alignment dataset
        """
        # Collect raw data
        raw_data = self.collect_raw_data(sources, filters)
        
        # Classify instructions
        classified_data = self.classify_instructions(raw_data)
        
        # Ensure diversity
        diverse_dataset = self.ensure_diversity(classified_data)
        
        return diverse_dataset

# Example Usage
def main():
    # Initialize alignment data constructor
    data_constructor = AlignmentDataConstructor()
    
    # Define data sources
    sources = [
        'academic_papers',
        'online_forums',
        'expert_interviews',
        'collaborative_platforms'
    ]
    
    # Construct alignment dataset
    alignment_dataset = data_constructor.construct_alignment_dataset(
        sources, 
        filters={
            'min_length': 50,
            'max_length': 500,
            'quality_threshold': 0.7
        }
    )
    
    print(alignment_dataset.head())
    print(f"Total diverse samples: {len(alignment_dataset)}")

if __name__ == "__main__":
    main()