import torch.optim as optim

# ... (SiameseNetwork definition from previous response)

# Hyperparameters (adjust these)
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# Loss function
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i in range(0, len(synthetic_data), batch_size):
        batch = synthetic_data[i:i + batch_size]

        text1_batch = [extract_text_features(item["text1"]) for item in batch]
        image1_batch = [extract_image_features(item["image1"]) for item in batch]
        text2_batch = [extract_text_features(item["text2"]) for item in batch]
        image2_batch = [extract_image_features(item["image2"]) for item in batch]
        preferences = torch.tensor([item["preference"] for item in batch], dtype=torch.float32).unsqueeze(1)

        # Convert lists of tensors to a single tensor (stack them)
        text1_batch = torch.cat(text1_batch, dim=0)
        image1_batch = torch.cat(image1_batch, dim=0)
        text2_batch = torch.cat(text2_batch, dim=0)
        image2_batch = torch.cat(image2_batch, dim=0)


        # Forward pass
        outputs = model(text1_batch, image1_batch, text2_batch, image2_batch)

        # Calculate loss
        loss = criterion(outputs, preferences)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save the trained model
torch.save(model.state_dict(), "preference_model.pth")