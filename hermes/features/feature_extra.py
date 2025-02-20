from transformers import BertTokenizer, BertModel
import torch
from torchvision import transforms

# Text Feature Extractor
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def extract_text_features(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)  # Average pooling

# Image Feature Extractor (using a simple transform for now)
image_transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Consistent size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

def extract_image_features(image):
  return image_transform(image).unsqueeze(0) # Add batch dimension


# Example usage:
text_features1 = extract_text_features(synthetic_data[0]["text1"])
image_features1 = extract_image_features(synthetic_data[0]["image1"])