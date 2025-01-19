import os
import requests
import zipfile
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import InceptionV3
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm

# Function to download and extract datasets
def download_and_extract_dataset(url, extract_path):
    if not os.path.exists(extract_path):
        os.makedirs(extract_path, exist_ok=True)
        file_path = os.path.join(extract_path, "dataset.zip")
        print(f"Downloading dataset from {url}...")
        response = requests.get(url, stream=True)
        with open(file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print("Download complete. Extracting files...")
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        os.remove(file_path)  # Clean up the zip file
        print(f"Dataset extracted to {extract_path}.")
    else:
        print(f"Dataset already exists at {extract_path}.")
    return extract_path

# Advanced image preprocessing function
def preprocess_images(images, img_shape, normalize=True, augment=False):
    """
    Preprocess the input images: normalize, resize, augment (optional).
    
    Parameters:
    - images (numpy array): Input image batch.
    - img_shape (tuple): Target shape for resizing.
    - normalize (bool): Whether to normalize the image data. Default is True.
    - augment (bool): Whether to apply data augmentation. Default is False.
    
    Returns:
    - images (numpy array): Preprocessed images.
    """
    # Convert to float32 if not already
    images = images.astype("float32")
    
    # Normalize images to [0, 1] range
    if normalize:
        images /= 255.0  # Standard normalization for pixel values
    
    # Ensure images have 4 dimensions: (batch_size, height, width, channels)
    if images.ndim == 3:  # If input is (batch_size, height, width)
        images = np.expand_dims(images, axis=-1)  # Convert to (batch_size, height, width, 1)
    elif images.ndim == 4:  # If input is already (batch_size, height, width, channels), no change
        pass
    else:
        raise ValueError(f"Invalid input shape: {images.shape}. Expected 3D or 4D array.")
    
    # Resize images to the target shape
    images = tf.image.resize(images, img_shape[:2])  # Resize to target height and width

    # Apply data augmentation if requested
    if augment:
        images = tf.image.random_flip_left_right(images)  # Horizontal flip
        images = tf.image.random_flip_up_down(images)    # Vertical flip
        images = tf.image.random_contrast(images, 0.2, 1.8) # Random contrast
    
    return images

# Function to load medical datasets
def load_medical_dataset(dataset_name):
    base_dir = os.path.expanduser("~/Downloads/MedicalDatasets/")  # Base directory to store datasets
    dataset_urls = {
        "nih_xray": "https://nihcc.app.box.com/v/ChestXray-NIHCC",
        "diabetic_retinopathy": "https://www.kaggle.com/c/diabetic-retinopathy-detection/data",
        "pneumonia_mnist": "https://medmnist.com/"

    }

    if dataset_name not in dataset_urls:
        raise ValueError("Dataset not supported. Available options: nih_xray, diabetic_retinopathy, pneumonia_mnist")

    extract_path = os.path.join(base_dir, dataset_name)
    download_and_extract_dataset(dataset_urls[dataset_name], extract_path)

    # Placeholder for loading datasets
    num_samples = 1000
    img_shape = (64, 64, 1)
    return preprocess_images(np.random.rand(num_samples, 64, 64), img_shape), np.random.randint(0, 2, num_samples)

# Build Generator Model
def build_generator(latent_dim, img_shape):
    model = tf.keras.Sequential([
        layers.Dense(256, activation="relu", input_dim=latent_dim),
        layers.Reshape((4, 4, 16)),  # Ensure this matches the output shape
        layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same", activation="relu"),
        layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding="same", activation="relu"),
        layers.Conv2DTranspose(32, (4, 4), strides=(2, 2), padding="same", activation="relu"),
        layers.Conv2D(img_shape[-1], (7, 7), activation="tanh", padding="same"),
    ])
    return model

# Build Discriminator Model
def build_discriminator(img_shape):
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=img_shape),  # Input layer explicitly defines the shape
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# FID Score Calculation
def calculate_fid(real_images, synthetic_images):
    model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    real_images = tf.image.resize(real_images, (299, 299))
    synthetic_images = tf.image.resize(synthetic_images, (299, 299))
    real_act = model.predict(real_images)
    synthetic_act = model.predict(synthetic_images)
    mu1, sigma1 = np.mean(real_act, axis=0), np.cov(real_act, rowvar=False)
    mu2, sigma2 = np.mean(synthetic_act, axis=0), np.cov(synthetic_act, rowvar=False)
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):  # Handle complex numbers
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

# Save Synthetic Dataset
def save_synthetic_dataset(X_synthetic, y_synthetic, dataset_name, path):
    dataset_dir = os.path.join(path, f'images_{dataset_name}')
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    for i in range(X_synthetic.shape[0]):
        img = (X_synthetic[i] * 255).astype(np.uint8)
        img_path = os.path.join(dataset_dir, f'image_{i}_{y_synthetic[i]}.png')
        tf.keras.preprocessing.image.save_img(img_path, img)

    print(f"Synthetic dataset saved at {dataset_dir}")

# Visualization of ROC Curve
def plot_roc_curve(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()

# Train GAN
def train_gan(generator, discriminator, dataset, epochs, batch_size, latent_dim):
    real_images, labels = dataset
    for epoch in range(epochs):
        idx = np.random.randint(0, real_images.shape[0], batch_size)
        real_batch = real_images[idx]
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        synthetic_batch = generator.predict(noise)
        labels_real = np.ones((batch_size, 1))
        labels_synthetic = np.zeros((batch_size, 1))
        d_loss_real = discriminator.train_on_batch(real_batch, labels_real)
        d_loss_fake = discriminator.train_on_batch(synthetic_batch, labels_synthetic)
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        labels_gan = np.ones((batch_size, 1))
        g_loss = discriminator.train_on_batch(generator.predict(noise), labels_gan)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | D Loss Real: {d_loss_real:.4f}, Fake: {d_loss_fake:.4f} | G Loss: {g_loss:.4f}")

# Main Execution
if __name__ == "__main__":
    dataset_name = input("Enter dataset name (nih_xray, diabetic_retinopathy, pneumonia_mnist): ")
    dataset = load_medical_dataset(dataset_name)
    latent_dim = 100
    img_shape = dataset[0][0].shape[1:]  # Assuming dataset is (images, labels)
    generator = build_generator(latent_dim, img_shape)
    discriminator = build_discriminator(img_shape)
    train_gan(generator, discriminator, dataset, epochs=5000, batch_size=32, latent_dim=latent_dim)
    real_images, labels = dataset
    synthetic_images = generator.predict(np.random.normal(0, 1, (len(real_images), latent_dim)))
    fid = calculate_fid(real_images, synthetic_images)
    print(f"FID Score: {fid:.4f}")
    save_directory = input("Enter the directory to save the synthetic dataset or press Enter to save in Downloads: ")
    if not save_directory:
        save_directory = os.path.join(os.path.expanduser("~"), "Downloads")
    save_synthetic_dataset(synthetic_images, labels, dataset_name, save_directory)
