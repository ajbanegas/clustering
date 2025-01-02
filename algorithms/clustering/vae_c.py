import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from common import NELEMS, SEED, get_closest_elems, hamming_distance, save_model, load_model
from matplotlib import pyplot as plt

LATENT_DIM = 16
KL_WEIGHT = .5

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))  # Sample noise
        return z_mean + tf.exp(.5 * z_log_var) * epsilon # 0.5

class VAE(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, kl_weight=0.01, **kwargs):
        super(VAE, self).__init__(**kwargs)
        # Encoder
        inputs = layers.Input(shape=(input_dim,))
        x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(32, activation="relu", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(20, activation="relu", kernel_initializer="he_normal")(x)
        x = layers.Dropout(.1)(x)
        self.z_mean = layers.Dense(latent_dim, name="z_mean", activation="linear")(x)
        self.z_log_var = layers.Dense(latent_dim, name="z_log_var", activation="linear")(x)
        z = Sampling()([self.z_mean, self.z_log_var])

        # Decoder
        latent_inputs = layers.Input(shape=(latent_dim,))
        x = layers.Dense(20, activation="relu", kernel_initializer="he_normal")(latent_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(32, activation="relu", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(128, activation="relu", kernel_initializer="he_normal")(x)
        x = layers.Dropout(.1)(x)
        outputs = layers.Dense(input_dim)(x) # , activation='sigmoid'

        self.encoder = tf.keras.Model(inputs, [self.z_mean, self.z_log_var, z], name="encoder")
        self.decoder = tf.keras.Model(latent_inputs, outputs, name="decoder")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight #tf.keras.backend.variable(kl_weight)

    def call(self, x):
        beta = .001 # .001
        z_mean, z_log_var, z = self.encoder(x)
        reconstructed = self.decoder(z)
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(x - reconstructed), axis=-1)  # MSE
        kl_loss = -0.5 * tf.keras.backend.sum(1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var), axis=-1)  # KL
        total_loss = tf.keras.backend.mean(reconstruction_loss + beta * kl_loss)
        self.add_loss(total_loss)
        return total_loss

    def get_config(self):
        config = super(VAE, self).get_config()
        config.update({
            "kl_weight": self.kl_weight,
            "latent_dim": self.latent_dim,
            "input_dim": self.input_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        kl_weight = config.pop("kl_weight")
        latent_dim = config.pop("latent_dim")
        input_dim = config.pop("input_dim")
        return cls(input_dim=input_dim, kl_weight=kl_weight, latent_dim=latent_dim, **config)


def get_classifier(n_clusters, X, dataset=""):
    input_dim = X.shape[1]
    latent_dim = LATENT_DIM
    best_model_path = f"temp-vae-{dataset}.keras"
    batch_size = 64
    epochs = 500
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.25, # .25
        patience=10, # 10
        min_lr=1e-27 # 1e-7
    )

    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        verbose=1, 
        patience=60,
        restore_best_weights=False
    )

    mckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor='val_loss',
        mode='min',
        verbose=1,
        save_best_only=True
    )

    vae = VAE(input_dim, latent_dim, kl_weight=KL_WEIGHT)
    vae.compile(optimizer=optimizer)
    history = vae.fit(X, X, 
                      epochs=epochs, 
                      batch_size=batch_size,
                      validation_split=.2,
                      callbacks=[es_callback, mckpt_callback, lr_callback] 
            )
    vae.load_weights(best_model_path)
    vae.save(f"models/vae-encoder-{dataset}.keras")

    #plot_loss(dataset, history)
    #plot_latent_space(dataset, X, vae, n_clusters)

    latent_space = vae.encoder.predict(X)[2]
    clf = KMeans(n_clusters=n_clusters, random_state=42) 
    return clf.fit(latent_space)
    
def classify(clf, df, X, query, dataset=""):
    query = np.nan_to_num(query)
    vae = tf.keras.models.load_model(f"models/vae-encoder-{dataset}.keras")

    ##### K-MEANS #####
    centroids = clf.cluster_centers_
    latent_space_query = vae.encoder.predict(query.reshape(1,-1))

    distances = [0] * len(centroids)
    for i, centroid in enumerate(centroids):
        distances[i] = hamming_distance(centroid, np.squeeze(latent_space_query[0]))

    closest_cluster = np.argmin(distances, axis=0)
    label = clf.labels_[closest_cluster]
    elems = np.where(label == clf.labels_)[0]

    if len(elems) > NELEMS:
        elems = get_closest_elems(df, elems, query, X)

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
    latent_representations = vae.encoder.predict(X)[2]

    pca = PCA(n_components=2)
    original_2d = pca.fit_transform(X)
    latent_2d = pca.fit_transform(latent_representations)

    #explained_variance_ratio = pca.explained_variance_ratio_
    #cumulative_variance = np.cumsum(explained_variance_ratio)
    #print('Explained Variance Ratio:', explained_variance_ratio)
    #print('Cumulative Variance:', cumulative_variance)

    clf = KMeans(n_clusters=n_clusters, random_state=42).fit(latent_representations)
    cluster_labels = clf.labels_
    #clf = DBSCAN(eps=.1, min_samples=10, n_jobs=-1, leaf_size=5).fit(latent_representations) #
    #cluster_labels = clf.labels_

    # Plot Original Space
    plt.figure(figsize=(12, 6))
    plt.suptitle(dataset)
    plt.subplot(1, 2, 1)
    plt.scatter(original_2d[:, 0], original_2d[:, 1], c=cluster_labels, cmap='viridis', s=10)
    #plt.colorbar(label="Cluster")
    plt.title("Original Space (PCA Reduced)")
    plt.xlabel("Original Dimension 1")
    plt.ylabel("PCA Dimension 2")

    # Plot Latent Space
    plt.subplot(1, 2, 2)
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=cluster_labels, cmap='viridis', s=10)
    plt.colorbar(label="Cluster")
    plt.title("Latent Space (PCA Reduced)")
    plt.xlabel("Latent Dimension 1")
    #plt.ylabel("Latent Dimension 2")

    plt.tight_layout()
    plt.savefig(f"images/space-{dataset}.png")
    plt.close()

