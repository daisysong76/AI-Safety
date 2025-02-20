import numpy as np
import random
from PIL import Image

def generate_synthetic_data(num_samples=1000):
    data =[]
    for _ in range(num_samples):
        # Text (simplified)
        text1 = random.choice(["cat", "dog", "bird", "fish"])
        text2 = random.choice(["cat", "dog", "bird", "fish"])

        # Image (placeholder - replace with actual image generation/loading)
        image1 = np.random.randint(0, 256, size=(64, 64, 3)).astype(np.uint8)  # Dummy image
        image2 = np.random.randint(0, 256, size=(64, 64, 3)).astype(np.uint8)  # Dummy image

        # Preference (random for now – replace with a real preference function)
        preference = 1 if random.random() > 0.5 else 0  # 1: text1/image1 preferred, 0: text2/image2 preferred

        data.append({"text1": text1, "image1": Image.fromarray(image1), "text2": text2, "image2": Image.fromarray(image2), "preference": preference})
    return data

synthetic_data = generate_synthetic_data()

# Example of saving a dummy image
synthetic_data["image1"].save("dummy_image.png")