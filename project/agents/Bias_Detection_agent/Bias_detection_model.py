import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import scipy.stats as stats

class BiasMeasurement:
    def __init__(self, model_name='bert-base-uncased'):
        # Load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def word_embedding_bias(self, word_pairs):
        """
        Measure bias in word embeddings using vector space analysis
        
        Args:
            word_pairs (list): List of tuples containing (word1, word2, protected_attribute)
        
        Returns:
            dict: Bias scores for each attribute
        """
        bias_scores = {}
        
        for attribute_group in set(pair[2] for pair in word_pairs):
            group_pairs = [pair for pair in word_pairs if pair[2] == attribute_group]
            
            # Compute embedding differences
            embedding_diffs = []
            for word1, word2, _ in group_pairs:
                emb1 = self.embedding_model.encode(word1)
                emb2 = self.embedding_model.encode(word2)
                embedding_diffs.append(np.linalg.norm(emb1 - emb2))
            
            # Compute statistical measures
            bias_scores[attribute_group] = {
                'mean_diff': np.mean(embedding_diffs),
                'std_diff': np.std(embedding_diffs),
                'skewness': stats.skew(embedding_diffs)
            }
        
        return bias_scores
    
    def context_bias_analysis(self, sentences):
        """
        Analyze contextual bias in sentences
        
        Args:
            sentences (list): List of sentences to analyze
        
        Returns:
            dict: Bias indicators for different contexts
        """
        # Tokenize and get model representations
        inputs = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Simple bias proxy using attention patterns
        attention_bias = outputs.last_hidden_state.mean(dim=[0, 2])
        
        return {
            'attention_variation': torch.std(attention_bias),
            'context_sensitivity': torch.norm(attention_bias)
        }
    
    def stereotype_bias_test(self, occupation_words, gender_words):
        """
        Test for stereotype associations between occupations and genders
        
        Args:
            occupation_words (list): List of occupation terms
            gender_words (list): List of gender terms
        
        Returns:
            dict: Stereotype association scores
        """
        associations = []
        
        for occupation in occupation_words:
            for gender in gender_words:
                # Compute embedding similarity
                occ_emb = self.embedding_model.encode(occupation)
                gender_emb = self.embedding_model.encode(gender)
                
                similarity = np.dot(occ_emb, gender_emb) / (
                    np.linalg.norm(occ_emb) * np.linalg.norm(gender_emb)
                )
                associations.append((occupation, gender, similarity))
        
        return sorted(associations, key=lambda x: x[2], reverse=True)

# Example usage demonstration
def main():
    bias_detector = BiasMeasurement()
    
    # Word embedding bias example
    word_pairs = [
        ('doctor', 'nurse', 'gender'),
        ('engineer', 'teacher', 'gender'),
        ('programmer', 'designer', 'gender')
    ]
    
    print("Word Embedding Bias:")
    print(bias_detector.word_embedding_bias(word_pairs))
    
    # Stereotype bias test
    occupations = ['doctor', 'nurse', 'engineer', 'teacher']
    genders = ['male', 'female']
    
    print("\nStereotype Associations:")
    print(bias_detector.stereotype_bias_test(occupations, genders))

if __name__ == '__main__':
    main()