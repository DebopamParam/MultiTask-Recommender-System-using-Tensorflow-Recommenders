import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

def load_data():
    """Loads and preprocesses the Movielens dataset."""
    ratings = tfds.load('movielens/latest-small-ratings', split="train")
    movies = tfds.load('movielens/latest-small-movies', split="train")

    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
        "user_rating": x["user_rating"],
    })
    movies = movies.map(lambda x: x["movie_title"])

    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)

    movie_titles = movies.batch(1_000)
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    return train, test, movies, unique_user_ids, unique_movie_titles

if __name__ == '__main__':
    train_data, test_data, all_movies, unique_users, unique_movies = load_data()
    print(f"Number of training examples: {len(list(train_data.as_numpy_iterator()))}")
    print(f"Number of test examples: {len(list(test_data.as_numpy_iterator()))}")
    print(f"Number of unique users: {len(unique_users)}")
    print(f"Number of unique movies: {len(unique_movies)}")
