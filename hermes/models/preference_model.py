# (Siamese Network)
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self, text_feature_dim, image_feature_dim):
        super(SiameseNetwork, self).__init__()
        self.text_branch = nn.Sequential(
            nn.Linear(text_feature_dim, 128),
            nn.ReLU()
        )
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 62 * 62, 128),  # Adjust based on image size
            nn.ReLU()
        )
        self.fc = nn.Linear(256, 1) # Combine features

    def forward(self, text1, image1, text2, image2):
        text_embedding1 = self.text_branch(text1)
        image_embedding1 = self.image_branch(image1)
        text_embedding2 = self.text_branch(text2)
        image_embedding2 = self.image_branch(image2)

        combined1 = torch.cat((text_embedding1, image_embedding1), dim=1)
        combined2 = torch.cat((text_embedding2, image_embedding2), dim=1)

        # You can use different ways to compare the combined features:
        # 1. Dot product similarity
        # similarity = F.cosine_similarity(combined1, combined2)
        # preference_score = torch.sigmoid(similarity)

        # 2. Difference and a linear layer
        diff = torch.abs(combined1 - combined2)
        preference_score = torch.sigmoid(self.fc(diff)) # Sigmoid for probability

        return preference_score


# Initialize the model (you'll need to determine the text and image feature dimensions)
model = SiameseNetwork(768, 3*64*64) # 768 is BERT's output dim, adjust appropriately