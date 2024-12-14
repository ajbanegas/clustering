import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from common import NELEMS, SEED, get_closest_elems, hamming_distance
from matplotlib import pyplot as plt

LATENT_DIM = 10

class VAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(input_dim,)),
            layers.Dense(128, activation='relu', kernel_initializer='zeros'),
            #layers.BatchNormalization(),
            #layers.Dense(64, activation='relu', kernel_initializer='zeros'),
            #layers.BatchNormalization(),
            layers.Dense(32, activation='relu', kernel_initializer='zeros'),
            #layers.Dropout(.15),
            layers.Dense(latent_dim * 2)
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(latent_dim,)),
            #layers.Dropout(.15),
            layers.Dense(32, activation='relu', kernel_initializer='zeros'),
            #layers.BatchNormalization(),
            #layers.Dense(64, activation='relu', kernel_initializer='zeros'),
            #layers.BatchNormalization(),
            layers.Dense(128, activation='relu', kernel_initializer='zeros'),
            layers.Dense(input_dim, activation='sigmoid')
        ])
        self.latent_dim = latent_dim

    def encode(self, x):
        z_params = self.encoder(x)
        mu, logvar = tf.split(z_params, num_or_size_splits=2, axis=1)
        return mu, logvar

    def reparametrize(self, mu, logvar):
        eps = tf.random.normal(shape=tf.shape(mu))
        return mu + eps * tf.exp(0.5 * logvar)

    def decode(self, z):
        return self.decoder(z)

    def call(self, x):
        self.mu, self.logvar = self.encode(x)
        z = self.reparametrize(self.mu, self.logvar)
        reconstructed = self.decode(z)
        return reconstructed, self.mu, self.logvar

    #def vae_loss(self, ytrue, ypred, beta = 1.0):
    #    reconstruction_loss = tf.reduce_mean(tf.square(tf.cast(ytrue, tf.float32) - tf.cast(ypred, tf.float32)), axis=1)
    #    kl_loss = -0.5 * tf.reduce_sum(1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar), axis=1)
    #    return tf.reduce_mean(reconstruction_loss + beta * kl_loss)

def get_classifier(n_clusters, X):
    input_dim = X.shape[1]
    latent_dim = LATENT_DIM
    best_model_path = 'vae.keras'
    batch_size = 128
    epochs = 80
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.01,
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

    vae = VAE(input_dim, latent_dim)
    vae.compile(loss='binary_crossentropy', optimizer=optimizer)
    history = vae.fit(X, X, 
                      epochs=epochs, 
                      batch_size=batch_size, 
                      validation_split=.2, 
                      callbacks=[lr_callback, es_callback, mckpt_callback]
            )
    vae.load_weights(best_model_path)

    #plot_loss('Enamine', history)
    #plot_latent_space('Enamine', X, vae, n_clusters)

    mu, _ = vae.encode(X)
    latent_representations = mu.numpy()
    latent_representations = np.vstack(latent_representations)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit(latent_representations)

def classify(clf, df, X, query):
    query = np.nan_to_num(query)
    vae = VAE(X.shape[1], LATENT_DIM)
    mu, _ = vae.encode(X)
    latent_representation = np.vstack(mu.numpy())

    centroids = clf.cluster_centers_
    distances = pairwise_distances(latent_representation, centroids, metric='euclidean')
    closest_cluster = np.argmin(distances, axis=1)
    label = clf.labels_[closest_cluster]
    elems = np.where(label == clf.labels_)[0]

    if len(elems) > NELEMS:
        elems = get_closest_elems(df, elems, query)

    return elems

def plot_loss(dataset, history):
    fig, axs = plt.subplots(1,1, figsize=(6, 5))
    axs.plot(history.history['loss'][5:])
    axs.plot(history.history['val_loss'][5:])
    plt.title(f'Model loss - {dataset}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train','val'], loc='upper right')
    plt.tight_layout()
    plt.savefig(f"images/loss-{dataset}.png")
    plt.close()

def plot_latent_space(dataset, X, vae, n_clusters):
    latent_representations, _ = vae.encode(X)
    latent_representations = latent_representations.numpy()

    pca = PCA(n_components=2)
    original_2d = pca.fit_transform(X)
    latent_2d = PCA(n_components=2).fit_transform(latent_representations)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(latent_representations)
    cluster_labels = kmeans.labels_

    # Plot Original Space
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(original_2d[:, 0], original_2d[:, 1], c=cluster_labels, cmap='viridis', s=10)
    plt.colorbar(label="Cluster")
    plt.title("Original Space (PCA Reduced)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    # Plot Latent Space
    plt.subplot(1, 2, 2)
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=cluster_labels, cmap='viridis', s=10)
    plt.colorbar(label="Cluster")
    plt.title("Latent Space (PCA Reduced)")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")

    plt.tight_layout()
    plt.savefig(f"images/space-{dataset}.png")
    plt.close()

