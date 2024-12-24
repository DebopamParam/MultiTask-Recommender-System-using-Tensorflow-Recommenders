# Multi-task Recommender System
### ***(Under Development)***

### ***[Click here to Open NoteBook to see Approach Code](https://github.com/DebopamParam/MultiTask-Recommender-System-using-Tensorflow-Recommenders/blob/main/Multi-Task%20Recommender.ipynb)***
![image](https://github.com/user-attachments/assets/1988bd0a-7857-4f95-8cee-cc4c2cd1e665)

This project implements a multi-task recommender system using TensorFlow Recommenders (TFRS). It's based on the TensorFlow Recommenders example for building a multi-objective recommender for Movielens, using both implicit (movie watches) and explicit signals (ratings).

## Project Structure

```
multitask_recommender/
├── config.yaml       # Configuration file for hyperparameters
├── data.py         # Script for loading and preprocessing the Movielens dataset
├── model.py        # Script defining the multi-task recommender model
├── train.py        # Script for training and evaluating the model
├── requirements.txt # List of project dependencies
└── README.md       # This file
```

## Configuration

The `config.yaml` file contains the following hyperparameters:

- `embedding_dimension`: Dimension of the embedding vectors for users and movies.
- `rating_weight`: Weight for the rating prediction loss.
- `retrieval_weight`: Weight for the retrieval task loss.
- `learning_rate`: Learning rate for the optimizer.
- `epochs`: Number of training epochs.
- `batch_size`: Batch size for training.

## Dependencies

The project dependencies are listed in `requirements.txt`. To install them, run:

```bash
pip install -r requirements.txt
```

## Running the project

To train and evaluate the model, run the `train.py` script:

```bash
python train.py
```

This script will:

1. Load the Movielens dataset.
2. Build the multi-task recommender model.
3. Train the model using the training data.
4. Evaluate the model on the test data, reporting the retrieval top-100 accuracy and the ranking Root Mean Squared Error (RMSE).

## Model Description

The model is a multi-task model that jointly optimizes for two tasks:

1. **Rating Prediction:** Predicts the explicit ratings given by users to movies.
2. **Retrieval:** Predicts which movies a user will interact with (implicit feedback).

The model consists of:

- User and movie embedding layers.
- A rating prediction network that takes user and movie embeddings as input.
- Two task-specific loss functions: Mean Squared Error for rating prediction and a retrieval loss.
- Weighted combination of the two losses to train the model jointly.

## Getting Started

1. Clone the repository.
2. Navigate to the `multitask_recommender` directory.
3. Install the dependencies: `pip install -r requirements.txt`.
4. Run the training script: `python train.py`.

## Further Improvements

- Experiment with different loss weights to balance the importance of the two tasks.
- Explore different model architectures for the rating prediction network.
- Incorporate additional user and movie features.
- Evaluate the model with different metrics.
