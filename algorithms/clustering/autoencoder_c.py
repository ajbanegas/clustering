import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from common import NELEMS, SEED, get_closest_elems, hamming_distance
from matplotlib import pyplot as plt


class Autoencoder(tf.keras.Model):
    def __init__(self, input_dim, embedding_dim):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dropout(.25), # BANEGAS
            layers.Dense(32, activation='relu'),
            layers.Dense(embedding_dim, activation='relu')
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dropout(.25), # BANEGAS
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(input_dim, activation='sigmoid') # sigmoid
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def get_classifier(n_clusters, X):
    input_dim = X.shape[1]
    embedding_dim = 16
    best_model_path = 'autoencoder.keras'

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.25,
        patience=5,
        min_lr=0.0001
    )

    es_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            mode='min', 
            verbose=1, 
            patience=20,
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
                              epochs=120,
                              batch_size=32, # 64
                              shuffle=True,
                              validation_split=0.2,
                              callbacks=[lr_callback, es_callback, mckpt_callback]
                              )
    autoencoder.load_weights(best_model_path)
    plot_loss('Enamine', history)

    encoder_model = autoencoder.encoder
    X_embeddings = encoder_model.predict(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit(X_embeddings)

"""
def classify(clf, df, X, query):
    # find the closest element
    query = np.nan_to_num(query)
    
    # calculate clusters' centroids excluding noise
    cluster_labels = np.unique(clf.labels_[clf.labels_ != -1])
    centroids = np.array([X[clf.labels_ == label].mean(axis=0) for label in cluster_labels])

    # calculate the distance between the query and all the centroids
    dist = np.zeros(centroids.shape[0])
    for i, cent in enumerate(centroids):
        dist[i] = hamming_distance(cent, query)
        
    #dist = hamming_distance(X, query)
    closest = np.argmin(dist)

    # identify the label of the new sample
    label = clf.labels_[closest]

    # find the other elements in the cluster
    elems = np.where(label == clf.labels_)[0]
    if len(elems) > NELEMS:
        elems = get_closest_elems(df, elems, query)

    return elems
"""

def classify(clf, df, X, query):
    query = np.nan_to_num(query)
    autoencoder = Autoencoder(X.shape[1], 16)
    embedding = autoencoder.encoder.predict(X)

    centroids = clf.cluster_centers_
    distances = pairwise_distances(embedding, centroids, metric='euclidean')
    closest_cluster = np.argmin(distances, axis=1)
    label = clf.labels_[closest_cluster]
    elems = np.where(label == clf.labels_)[0]

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

