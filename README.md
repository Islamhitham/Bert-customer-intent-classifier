# Customer Intent Recognition with BERT

This project fine-tunes a BERT model to classify customer intent from text. It includes scripts for training, inference, and benchmarking.

## Setup

Ensure you have the necessary libraries installed.

```bash
pip install transformers datasets torch scikit-learn evaluate
```

## 1. Training

To train the model on the `customer_data` dataset:

```bash
python train.py
```

This will:
- Download the dataset.
- Fine-tune `bert-base-uncased` for 4 epochs.
- Save the model to `./emotion_bert_model`.
- Save training logs and checkpoints to `./results`.

## 2. Inference

To predict the emotion of a text string:

```bash
python inference.py "I want to buy this piece"
```

Output:
```
--- Prediction Result ---
Text: I want to buy this piece
Emotion: purchase_intent
Confidence: 0.9985
...
```

## 3. Benchmarking

To measure the latency and throughput of the model:

```bash
python benchmark.py
```

This will run 1000 inference calls and report P50, P95, P99 latency and requests per second.

## Project Structure

- `train.py`: Main training script.
- `inference.py`: Script for single-sample prediction.
- `benchmark.py`: Script for performance testing.
- `emotion_bert_model/`: Directory where the trained model is saved (created after training).
