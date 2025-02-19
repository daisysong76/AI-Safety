import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
from transformers import BertTokenizer, BertModel
from torchvision import transforms

# ... (Feature extraction functions and BayesianSiameseNetwork from previous responses)

# 1. Create a Committee of Models:
num_committee_members = 5  # Number of models in the committee
committee = [BayesianSiameseNetwork(768, 3*64*64, dropout_rate=0.2) for _ in range(num_committee_members)]

# Initialize optimizers for each model
optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in committee] # Adjust learning rate

# ... (Training loop - train each model in the committee separately)

# 2. Query-by-Committee:
def query_by_committee(unlabeled_data, num_to_query):
    disagreement_scores = []
    for data_point in unlabeled_data:
        predictions = []
        for model in committee:
            text1 = extract_text_features(data_point["text1"])
            image1 = extract_image_features(data_point["image1"])
            text2 = extract_text_features(data_point["text2"])
            image2 = extract_image_features(data_point["image2"])
            prediction = model(text1, image1, text2, image2).detach().numpy()
            predictions.append(prediction)

        predictions = np.array(predictions).squeeze()
        # Calculate disagreement (e.g., variance, entropy)
        disagreement = np.var(predictions) # Variance as disagreement measure
        disagreement_scores.append(disagreement)

    indices_to_query = np.argsort(disagreement_scores)[-num_to_query:] # Get indices of largest disagreements
    return [unlabeled_data[i] for i in indices_to_query] # Return the actual data points


# Example usage:
unlabeled_data = generate_synthetic_data(num_samples=50) # Example unlabeled data
num_queries = 10
queried_data = query_by_committee(unlabeled_data, num_queries)

# Now get human feedback for the 'queried_data' and add it to the training set.
# Retrain the committee models with the new data.