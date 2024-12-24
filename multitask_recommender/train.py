import tensorflow as tf
from multitask_recommender.model import MovielensModel
from multitask_recommender.data import load_data
import yaml

def train_model():
    """Loads data, builds, trains, and evaluates the multi-task recommender model."""
    train_data, test_data, movies, unique_user_ids, unique_movie_titles = load_data()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    embedding_dimension = config['embedding_dimension']
    rating_weight = config['rating_weight']
    retrieval_weight = config['retrieval_weight']
    learning_rate = config['learning_rate']
    epochs = config['epochs']
    batch_size = config['batch_size']

    model = MovielensModel(unique_user_ids, unique_movie_titles, embedding_dimension, rating_weight, retrieval_weight)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate))

    cached_train = train_data.shuffle(100_000).batch(batch_size).cache()
    cached_test = test_data.batch(batch_size // 2).cache()

    model.fit(cached_train, epochs=epochs)
    metrics = model.evaluate(cached_test, return_dict=True)

    print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
    print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")

if __name__ == '__main__':
    train_model()
