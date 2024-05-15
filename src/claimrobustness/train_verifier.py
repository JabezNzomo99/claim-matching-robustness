# Save model and use it in selection
import argparse
import configparser
import os
from pathlib import Path
from claimrobustness import utils
from sklearn.metrics import classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
import random
import numpy as np
import pandas as pd
import time

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


def tokenize(tokenizer, sentences, labels, max_length):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        sent = utils.clean_tweet(sent)
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=max_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
        )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict["input_ids"])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict["attention_mask"])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels


def run():
    parser = argparse.ArgumentParser(
        description="Create a dataset for the verifier model"
    )
    parser.add_argument(
        "experiment_path",
        help="Path where config lies",
        type=str,
    )

    # Parse the arguments
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(os.path.join(args.experiment_path, "config.ini"))
    dataset_dir = config["data"].get("dataset_dir")

    # Check whether train verifier dataset exists
    if (
        Path(dataset_dir).is_dir() == False
        or (Path.cwd() / dataset_dir / "train_verifier_dataset.csv").exists() == False
    ):
        raise ValueError(
            f"Dataset {dataset_dir} not found, run create-verifer-dataset first"
        )

    # Load the dataset
    train_data = utils.load_verifier_data(
        dataset_path=os.path.join(dataset_dir, "train_verifier_dataset.csv")
    )
    dev_data = utils.load_verifier_data(
        dataset_path=os.path.join(dataset_dir, "dev_verifier_dataset.csv")
    )
    test_data = utils.load_verifier_data(
        dataset_path=os.path.join(dataset_dir, "test_verifier_dataset.csv")
    )

    print("Train data shape: ", train_data.shape)
    print("Dev data shape: ", dev_data.shape)
    print("Test data shape: ", test_data.shape)

    train_sentences, train_targets = utils.combine_features(train_data)
    dev_sentences, dev_targets = utils.combine_features(dev_data)
    test_sentences, test_targets = utils.combine_features(test_data)

    model_string = config["model"].get("model_string")
    max_length = int(config["training"].getint("max_length"))
    tokenizer = AutoTokenizer.from_pretrained(model_string)

    train_input_ids, train_attention_masks, train_labels = tokenize(
        tokenizer, train_sentences, train_targets, max_length
    )
    dev_input_ids, dev_attention_masks, dev_labels = tokenize(
        tokenizer, dev_sentences, dev_targets, max_length
    )
    test_input_ids, test_attention_masks, test_labels = tokenize(
        tokenizer, test_sentences, test_targets, max_length
    )

    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    dev_dataset = TensorDataset(dev_input_ids, dev_attention_masks, dev_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

    batch_size = config["training"].getint("batch_size")
    train_dataloader = DataLoader(
        train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size
    )
    dev_dataloader = DataLoader(
        dev_dataset, sampler=SequentialSampler(dev_dataset), batch_size=batch_size
    )
    test_dataloader = DataLoader(
        test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_string, num_labels=config["training"].getint("num_labels")
    )
    device = utils.get_device()
    model.to(device)

    epochs = config["training"].getint("epochs")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"].getfloat("lr"),
        eps=config["training"].getfloat("eps"),
    )

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    training_stats = []
    # Measure the total training time for the whole run.
    total_t0 = time.time()
    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.
        print("")
        print("======== Epoch {:} / {:} ========".format(epoch_i + 1, epochs))
        print("Training...")
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_train_loss = 0
        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # Pick an interval on which to print progress updates.
        update_interval = utils.good_update_interval(
            total_iters=len(train_dataloader), num_desired_updates=10
        )

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update.
            if (step % update_interval) == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = utils.format_time(time.time() - t0)

                # Report progress.
                print(
                    "  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.".format(
                        step, len(train_dataloader), elapsed
                    )
                )

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # In PyTorch, calling `model` will in turn call the model's `forward`
            # function and pass down the arguments. The `forward` function is
            # documented here:
            # https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
            # The results are returned in a results object, documented here:
            # https://huggingface.co/transformers/main_classes/output.html#transformers.modeling_outputs.SequenceClassifierOutput
            # Specifically, we'll get the loss (because we provided labels) and the
            # "logits"--the model outputs prior to activation.
            result = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                return_dict=True,
            )

            logits = result.logits
            # loss = result.loss

            criterion = torch.nn.CrossEntropyLoss(reduction="mean")
            loss = criterion(logits, b_labels)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = utils.format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_loss = 0

        predictions, true_labels = [], []

        # Evaluate data for one epoch
        for batch in dev_dataloader:

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():

                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                result = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                    return_dict=True,
                )

            # Get the loss and "logits" output by the model. The "logits" are the
            # output values prior to applying an activation function like the
            # softmax.
            logits = result.logits

            # Calculate the loss without applying class weights
            criterion = torch.nn.CrossEntropyLoss(reduction="mean")
            loss = criterion(logits, b_labels)

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Calculate the accuracy for this batch of test sentences.

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        # Measure validation accuracy...

        # Combine the results across all batches.
        flat_predictions = np.concatenate(predictions, axis=0)
        flat_true_labels = np.concatenate(true_labels, axis=0)

        # For each sample, pick the label (0, 1, or 2) with the highest score.
        predicted_labels = np.argmax(flat_predictions, axis=1).flatten()

        # Calculate the validation accuracy.
        val_accuracy = (predicted_labels == flat_true_labels).mean()

        # Report the final accuracy for this validation run.
        print("  Accuracy: {0:.2f}".format(val_accuracy))

        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(dev_dataloader)

        # Measure how long the validation run took.
        validation_time = utils.format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                "epoch": epoch_i + 1,
                "Training Loss": avg_train_loss,
                "Valid. Loss": avg_val_loss,
                "Valid. Accur.": val_accuracy,
                "Training Time": training_time,
                "Validation Time": validation_time,
            }
        )

    print("")
    print("Training complete!")
    print(
        "Total training took {:} (h:mm:ss)".format(
            utils.format_time(time.time() - total_t0)
        )
    )

    # Save the training stats
    df_stats = pd.DataFrame(data=training_stats)
    # Use the 'epoch' as the row index.
    df_stats = df_stats.set_index("epoch")
    df_stats.to_csv(os.path.join(args.experiment_path, "training_stats.csv"))
    utils.plot_training_stats(df_stats, args.experiment_path)

    # Prediction on test set
    model.eval()
    # Tracking variables
    predictions, true_labels = [], []
    # Predict
    for batch in test_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and
        # speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            result = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                return_dict=True,
            )

    logits = result.logits
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to("cpu").numpy()
    # Store predictions and true labels
    predictions.append(logits)
    true_labels.append(label_ids)

    # Combine the results across all batches.
    flat_predictions = np.concatenate(predictions, axis=0)
    # For each sample, pick the label (0 or 1) with the higher score.
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    # Combine the correct labels for each batch into a single list.
    flat_true_labels = np.concatenate(true_labels, axis=0)

    accuracy = (flat_predictions == flat_true_labels).mean()
    print("Model Test Accuracy: {:.3f}".format(accuracy))
    print(
        classification_report(
            flat_true_labels,
            flat_predictions,
        )
    )

    # Save the model
    model.save_pretrained(args.experiment_path)
    print(
        f"Finished training verifier model {model_string} and saved output to {args.experiment_path} "
    )


if __name__ == "__main__":
    run()
