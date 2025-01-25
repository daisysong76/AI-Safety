from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from config import Config

import wandb
from transformers import Trainer

# Initialize WandB
wandb.init(project="bias-detection", name="fine-tune-LLM")

# Define WandB callback
class WandbCallback:
    def __call__(self, trainer, logs):
        wandb.log(logs)

def fine_tune_model():
    # Load dataset
    dataset = load_dataset("csv", data_files={"train": Config.PROCESSED_DATA_DIR + "cleaned_data.csv"})

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(Config.BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=Config.FINE_TUNED_MODEL_DIR,
        evaluation_strategy="steps",
        save_steps=10,
        logging_dir="./logs",
        per_device_train_batch_size=4,
        num_train_epochs=3,
    )

    # Train model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        callbacks=[WandbCallback()],
    )
    trainer.train()

if __name__ == "__main__":
    fine_tune_model()


# can use mlflow
# pip install mlflow

# Update fine_tune.py:
# import mlflow
# Log model parameters and metrics
# mlflow.start_run()
# mlflow.log_param("epochs", 3)
# mlflow.log_param("batch_size", 4)
# mlflow.log_metric("accuracy", 0.92)

# # Save the trained model
# model.save_pretrained("./models/fine_tuned/")
# mlflow.end_run()