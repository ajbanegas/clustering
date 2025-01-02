import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from common import NELEMS, SEED, get_closest_elems, hamming_distance, save_model, load_model
from matplotlib import pyplot as plt

EMBEDDING_DIM = 16

class Autoencoder(tf.keras.Model):
    def __init__(self, input_dim, embedding_dim, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        # Encoder
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(20, activation='relu'),
            layers.Dropout(.1), # .25
            layers.Dense(embedding_dim, activation='relu')
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(20, activation='relu'),
            #layers.Dropout(.1), # .25
            layers.Dense(32, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(input_dim) # sigmoid
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):
        config = super(Autoencoder, self).get_config()
        config.update({
            "embedding_dim": self.embedding_dim,
            "input_dim": self.input_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        embedding_dim = config.pop("embedding_dim")
        input_dim = config.pop("input_dim")
        return cls(input_dim=input_dim, embedding_dim=embedding_dim, **config)



def get_classifier(n_clusters, X, dataset=""):
    input_dim = X.shape[1]
    embedding_dim = EMBEDDING_DIM
    best_model_path = f"temp-autoencoder-{dataset}.keras"

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.25,
        patience=5,
        min_lr=1e-10
    )

    es_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            mode='min', 
            verbose=1, 
            patience=40,
            restore_best_weights=False
    )

    mckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor='val_loss',
            mode='min',
            verbose=1,
            save_best_only=True
    )

    autoencoder = Autoencoder(input_dim, embedding_dim)
    autoencoder.compile(optimizer='adam', loss='mae') # rmsprop, mse
    history = autoencoder.fit(X, X,
                              epochs=150,
                              batch_size=64, # 64
                              shuffle=True,
                              validation_split=0.2,
                              callbacks=[lr_callback, es_callback, mckpt_callback]
                              )
    autoencoder.load_weights(best_model_path)
    autoencoder.save(f"models/autoencoder-encoder-{dataset}.keras")
    #plot_loss('Drugbank', history)

    encoder_model = autoencoder.encoder
    X_embeddings = encoder_model.predict(X)

    clf = KMeans(n_clusters=n_clusters, random_state=42)
    return clf.fit(X_embeddings)

def classify(clf, df, X, query, dataset=""):
    query = np.nan_to_num(query)
    autoencoder = tf.keras.models.load_model(f"models/autoencoder-encoder-{dataset}.keras")

    ##### K-MEANS #####
    centroids = clf.cluster_centers_
    embedding_query = autoencoder.encoder.predict(query.reshape(1,-1))

    distances = [0] * len(centroids)
    for i, centroid in enumerate(centroids):
        distances[i] = hamming_distance(centroid, np.squeeze(embedding_query[0]))

    closest_cluster = np.argmin(distances, axis=0)
    label = clf.labels_[closest_cluster]
    elems = np.where(label == clf.labels_)[0]


    """
    autoencoder = Autoencoder(X.shape[1], EMBEDDING_DIM)

    cluster_labels = np.unique(clf.labels_[clf.labels_ != -1])
    centroids = np.array([X[clf.labels_ == label].mean(axis=0) for label in cluster_labels])
    #centroids = clf.cluster_centers_
    embedding_centroids = autoencoder.encoder.predict(centroids)
    embedding_query = autoencoder.encoder.predict(query.reshape(1,-1))
    embedding_query = np.squeeze(embedding_query[0])

    distances = []
    for centroid in embedding_centroids:
        distances.append(hamming_distance(centroid, embedding_query))

    closest_cluster = np.argmin(distances, axis=0)
    label = clf.labels_[closest_cluster]
    elems = np.where(label == clf.labels_)[0]
    """

    if len(elems) > NELEMS:
        elems = get_closest_elems(df, elems, query)

    return elems

def plot_loss(dataset, history):
    fig, axs = plt.subplots(1,1, figsize=(6, 5))
    axs.plot(history.history['loss'])
    axs.plot(history.history['val_loss'])
    plt.title(f'Model loss - {dataset}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper right')
    plt.tight_layout()
    plt.savefig(f"images/loss-{dataset}.png")
    plt.close()
