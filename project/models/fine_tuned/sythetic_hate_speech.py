import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import random
from datetime import datetime
import json
import re

class ContentSafetyDataGenerator:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Define content categories and their associated weights
        self.categories = {
            0: "safe",
            1: "harassment",
            2: "hate_speech",
            3: "self_harm",
            4: "sexual",
            5: "violence"
        }
        
        # Templates for different content types
        self._initialize_templates()

    def _initialize_templates(self):
        """Initialize templates for different content categories"""
        self.templates = {
            "safe": [
                "I really enjoyed {positive_activity} today!",
                "The weather is {weather_condition} for {outdoor_activity}.",
                "Just finished reading {book_topic} book about {academic_subject}.",
                "Had a great time {social_activity} with friends.",
                "Learning about {academic_subject} is so interesting!",
                "Looking forward to {future_event} next week.",
                "The new {product_type} I bought works perfectly.",
                "Thanks for sharing your thoughts about {discussion_topic}.",
                "I appreciate your help with {task_type}.",
                "Let's discuss {professional_topic} in the meeting."
            ],
            "harassment": [
                "You're so {negative_trait} at {activity}.",
                "Nobody wants to {activity} with you because {negative_reason}.",
                "Everyone knows you're {negative_trait}.",
                "Why don't you just {negative_suggestion}?",
                "You'll never succeed at {activity}."
            ],
            "hate_speech": [
                "All {group} are {negative_trait}.",
                "{group} should be {negative_action}.",
                "We don't want {group} in our {location}.",
                "{group} are ruining our {social_concept}.",
                "No {group} allowed in {location}."
            ],
            "self_harm": [
                "I feel like {negative_emotion} all the time.",
                "Nobody would care if I {negative_action}.",
                "I can't handle {situation} anymore.",
                "Everything would be better if I {negative_action}.",
                "I'm thinking about {harmful_action}."
            ],
            "sexual": [
                "[CENSORED TEMPLATE]",
                "[CENSORED TEMPLATE]",
                "[CENSORED TEMPLATE]",
                "[CENSORED TEMPLATE]",
                "[CENSORED TEMPLATE]"
            ],
            "violence": [
                "I want to {aggressive_action} them.",
                "They deserve to be {aggressive_action}.",
                "Someone should {aggressive_action} them all.",
                "I'm going to {aggressive_action} next time.",
                "Let's {aggressive_action} them."
            ]
        }
        
        # Template filling dictionaries
        self.template_fillers = {
            "positive_activity": ["cooking", "gardening", "painting", "exercising", "studying", "writing"],
            "weather_condition": ["perfect", "great", "wonderful", "ideal", "amazing"],
            "outdoor_activity": ["hiking", "jogging", "photography", "picnicking", "gardening"],
            "book_topic": ["fascinating", "interesting", "educational", "inspiring", "thought-provoking"],
            "academic_subject": ["history", "science", "mathematics", "literature", "philosophy"],
            "social_activity": ["having lunch", "playing games", "watching movies", "chatting", "working"],
            "future_event": ["the conference", "the workshop", "the meeting", "the celebration", "the gathering"],
            "product_type": ["software", "book", "tool", "device", "application"],
            "discussion_topic": ["the project", "the proposal", "the ideas", "the research", "the plans"],
            "task_type": ["the analysis", "the research", "the development", "the design", "the implementation"],
            "professional_topic": ["strategy", "development", "innovation", "improvement", "collaboration"],
            
            # Negative templates - keeping it mild for demonstration
            "negative_trait": ["bad", "terrible", "awful", "poor", "useless"],
            "negative_reason": ["you're incompetent", "you're not qualified", "you're not good enough", "you're unprepared", "you're unsuitable"],
            "negative_suggestion": ["give up", "stop trying", "quit", "leave", "go away"],
            "negative_emotion": ["sad", "hopeless", "worthless", "empty", "lost"],
            "negative_action": ["disappeared", "gone", "left", "vanished", "away"],
            "situation": ["the pressure", "the stress", "the responsibility", "the expectations", "the demands"],
            "harmful_action": ["[REMOVED]", "[REMOVED]", "[REMOVED]", "[REMOVED]", "[REMOVED]"],
            "aggressive_action": ["[REMOVED]", "[REMOVED]", "[REMOVED]", "[REMOVED]", "[REMOVED]"],
            "group": ["[REMOVED]", "[REMOVED]", "[REMOVED]", "[REMOVED]", "[REMOVED]"],
            "location": ["here", "this place", "this area", "this community", "this group"],
            "social_concept": ["society", "community", "culture", "values", "traditions"]
        }

    def generate_text(self, category: str) -> str:
        """Generate text for a specific category"""
        template = random.choice(self.templates[category])
        
        # Fill in template placeholders
        for key in re.findall(r'\{(\w+)\}', template):
            if key in self.template_fillers:
                template = template.replace(
                    f'{{{key}}}',
                    random.choice(self.template_fillers[key])
                )
        
        return template

    def generate_dataset(self, 
                        num_samples: int = 1000,
                        class_distribution: Dict[int, float] = None) -> pd.DataFrame:
        """
        Generate synthetic dataset with specified number of samples
        and class distribution
        """
        if class_distribution is None:
            # Default distribution (imbalanced, as in real-world scenarios)
            class_distribution = {
                0: 0.7,  # safe
                1: 0.06, # harassment
                2: 0.06, # hate_speech
                3: 0.06, # self_harm
                4: 0.06, # sexual
                5: 0.06  # violence
            }
        
        # Validate distribution
        assert sum(class_distribution.values()) == 1.0, "Class distribution must sum to 1"
        
        data = []
        for _ in range(num_samples):
            # Select category based on distribution
            category_id = np.random.choice(
                list(class_distribution.keys()),
                p=list(class_distribution.values())
            )
            category = self.categories[category_id]
            
            # Generate text
            text = self.generate_text(category)
            
            # Add metadata
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            data.append({
                "text": text,
                "label": category_id,
                "category": category,
                "timestamp": timestamp,
                "length": len(text)
            })
        
        return pd.DataFrame(data)

    def save_dataset(self, 
                    df: pd.DataFrame, 
                    file_path: str = "synthetic_content_safety_data.csv"):
        """Save dataset to file"""
        df.to_csv(file_path, index=False)
        print(f"Dataset saved to {file_path}")
        
        # Save dataset statistics
        stats = {
            "total_samples": len(df),
            "class_distribution": df["label"].value_counts().to_dict(),
            "avg_text_length": df["length"].mean(),
            "min_text_length": df["length"].min(),
            "max_text_length": df["length"].max()
        }
        
        stats_file = file_path.replace(".csv", "_stats.json")
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Dataset statistics saved to {stats_file}")

def main():
    # Initialize generator
    generator = ContentSafetyDataGenerator(seed=42)
    
    # Generate dataset with custom distribution
    class_distribution = {
        0: 0.7,   # safe
        1: 0.06,  # harassment
        2: 0.06,  # hate_speech
        3: 0.06,  # self_harm
        4: 0.06,  # sexual
        5: 0.06   # violence
    }
    
    # Generate different dataset sizes
    sizes = [1000, 5000, 10000]
    for size in sizes:
        df = generator.generate_dataset(
            num_samples=size,
            class_distribution=class_distribution
        )
        
        # Save dataset
        generator.save_dataset(df, f"synthetic_content_safety_data_{size}.csv")
        
        # Print sample from each category
        print(f"\nSample texts from {size} dataset:")
        for category_id, category in generator.categories.items():
            sample = df[df["label"] == category_id]["text"].iloc[0]
            print(f"{category}: {sample}")

if __name__ == "__main__":
    main()


#     Here are several relevant public datasets you could use for content safety classification:

# 1. **Jigsaw Toxic Comment Classification Dataset**
# - Source: Kaggle (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
# - Contains ~160k comments labeled for toxicity, severe toxicity, obscenity, threat, insult, and identity hate
# - One of the most widely used datasets for content moderation
# - High quality and well-balanced

# 2. **HateXplain Dataset**
# - Source: https://github.com/hate-alert/HateXplain
# - ~20k posts from Twitter and Gab
# - Annotated for hate speech, offensive content, and normal content
# - Includes word-level annotations and target communities
# - Particularly useful because it includes explanations for classifications

# 3. **Twitter Hate Speech Dataset**
# - Source: https://huggingface.co/datasets/hate_speech18
# - Contains ~100k tweets with binary labels (hate speech or not)
# - Includes different types of hate speech and offensive language

# 4. **Wikipedia Detox Dataset**
# - Source: https://meta.wikimedia.org/wiki/Research:Detox
# - Contains comments from Wikipedia talk pages
# - Labeled for personal attacks and toxicity
# - High-quality annotations from multiple reviewers

# 5. **Civil Comments Dataset**
# - Source: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data
# - ~2M comments with toxicity scores
# - Includes identity attributes to help measure unintended bias
# - Very comprehensive but requires Kaggle account

# 6. **Reddit Hate Speech Dataset**
# - Available on Hugging Face: https://huggingface.co/datasets/hate_speech_offensive
# - Contains Reddit comments labeled for hate speech and offensive content
# - Good for capturing platform-specific content patterns

# 7. **OLID (Offensive Language Identification Dataset)**
# - Source: https://sites.google.com/site/offensevalsharedtask/olid
# - Hierarchical labeling scheme
# - Contains tweets labeled for offensive language, categorized by type and target

# For getting started, I would recommend:

# 1. **For smaller scale testing/development:**
#    - Start with HateXplain or OLID
#    - More manageable size
#    - Well-documented and clean

# 2. **For production models:**
#    - Jigsaw Toxic Comment or Civil Comments datasets
#    - Larger scale
#    - More diverse content types
#    - Better representation of real-world scenarios

# Important considerations:
# - Many datasets require accepting terms of use
# - Some may require academic affiliation
# - Consider combining multiple datasets for better coverage
# - Check licensing terms before commercial use
# - Be aware of potential biases in the datasets

# Would you like me to provide more specific information about any of these datasets or help you evaluate which would be best for your specific use case?