import logging
import traceback
from datetime import datetime

import numpy as np
import os
from datasets import load_dataset
from sentence_transformers.losses import MSELoss
from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import CosineSimilarityLoss
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.evaluation import (
    MSEEvaluator,
)

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
logger = logging.getLogger(__name__)

###############################################################################
# 1. Define model names and hyperparameters
###############################################################################
teacher_model_name = "sentence-transformers/all-mpnet-base-v2"
student_model_name = "sentence-transformers/all-mpnet-base-v2"
student_max_seq_length = 256
train_batch_size = 64
inference_batch_size = 64  # Batch size at inference
num_train_epochs = 20
num_evaluation_steps = 500  # Evaluate every 500 steps

# CSV paths (they must each have columns "original_query" and "perturbed_query")
train_path = "train_perturbed_queries_full_train.csv"
eval_path = "train_perturbed_queries_full_eval.csv"

data_dir = "/data/kebl7383/claimrobustness"
output_dir = "output/original-perturbed-full-" + datetime.now().strftime("%Y-%m-%d_%H")
# Merge the two paths
output_dir = os.path.join(data_dir, output_dir)
os.makedirs(output_dir, exist_ok=True)

###############################################################################
# 2. Load (or create) the SentenceTransformer model
###############################################################################
# 1a. Here we define our SentenceTransformer teacher model.
teacher_model = SentenceTransformer(teacher_model_name)
# If we want, we can limit the maximum sequence length for the model
# teacher_model.max_seq_length = 128
logging.info(f"Teacher model: {teacher_model}")

# 1b. Here we define our SentenceTransformer student model. If not already a Sentence Transformer model,
# it will automatically create one with "mean" pooling.
student_model = SentenceTransformer(student_model_name)
# If we want, we can limit the maximum sequence length for the model
student_model.max_seq_length = student_max_seq_length
logging.info(f"Student model: {student_model}")

###############################################################################
# 3. Load training and evaluation data from CSV
###############################################################################
train_dataset = load_dataset("csv", data_files=train_path, split="train")
eval_dataset = load_dataset("csv", data_files=eval_path, split="train")
logging.info(f"Loaded train dataset: {train_dataset}")
logging.info(f"Loaded eval dataset:  {eval_dataset}")


###############################################################################
# 4. Convert them to the format SentenceTransformerTrainer expects
#    We map each row into {"text1": ..., "text2": ..., "label": ...}
#    We'll label all original_query vs. perturbed_query pairs with 1.0
#    indicating they should be similar.
###############################################################################
def prepare_dataset(batch):
    return {
        "original_query": batch["original_query"],
        "perturbed_query": batch["perturbed_query"],
        "label": teacher_model.encode(
            batch["original_query"],
            batch_size=inference_batch_size,
            show_progress_bar=False,
        ),
    }


train_dataset = train_dataset.map(
    prepare_dataset, batched=True, remove_columns=train_dataset.column_names
)

eval_dataset = eval_dataset.map(
    prepare_dataset, batched=True, remove_columns=eval_dataset.column_names
)

###############################################################################
# 5. Define the training loss (CosineSimilarityLoss to bring embeddings close)
###############################################################################
# MSELoss (https://sbert.net/docs/package_reference/sentence_transformer/losses.html#mseloss) needs one text columns and one
# column with embeddings from the teacher model
train_loss = MSELoss(model=student_model)

###############################################################################
# 6. (Optional) Define an evaluator to measure similarity on the dev set
###############################################################################
eval_texts1 = eval_dataset["original_query"]
eval_texts2 = eval_dataset["perturbed_query"]
eval_labels = eval_dataset["label"]

evaluator = MSEEvaluator(
    source_sentences=eval_dataset["original_query"],
    target_sentences=eval_dataset["perturbed_query"],
    teacher_model=teacher_model,
    batch_size=inference_batch_size,
)

###############################################################################
# 7. Define training arguments for SentenceTransformerTrainer
###############################################################################
args = SentenceTransformerTrainingArguments(
    # Required parameter:
    output_dir=output_dir,
    # Optional training parameters:
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=train_batch_size,
    warmup_ratio=0.1,
    bf16=True,  # Set to True if you have a GPU that supports BF16
    learning_rate=2e-5,  # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=num_evaluation_steps,
    save_strategy="steps",
    save_steps=num_evaluation_steps,
    save_total_limit=2,
    logging_steps=10,
    run_name="original-perturbed",  # Will be used in W&B if `wandb` is installed
)

###############################################################################
# 8. Create the trainer and train
###############################################################################
trainer = SentenceTransformerTrainer(
    model=student_model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=train_loss,
    evaluator=evaluator,
)

trainer.train()

# Perform and log the final evaluation
final_mse = evaluator(student_model)
logging.info(f"Final evaluation MSE: {final_mse:.5f}")

###############################################################################
# 9. Save the trained model locally
###############################################################################
final_output_dir = f"{output_dir}/final"
student_model.save(final_output_dir)
logging.info(f"Model saved to: {final_output_dir}")

###############################################################################
# 10. (Optional) push to Hugging Face Hub
###############################################################################
# model_name_for_hub = "my-original-perturbed-model"
# try:
#     model.push_to_hub(model_name_for_hub)
#     logging.info(f"Model pushed to the Hugging Face Hub as: {model_name_for_hub}")
# except Exception as e:
#     logging.error(f"Error uploading model to HF Hub:\n{traceback.format_exc()}")
#     logging.info(
#         "You can log in first via `huggingface-cli login` "
#         f"and then manually push the model at {final_output_dir}."
#     )
