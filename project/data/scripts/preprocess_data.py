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