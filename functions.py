
import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

def meanshift_segmentation(image_path):
    # Convertir l'image en RGB
    image=cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Appliquer un flou médian pour réduire le bruit
    image_filtered = cv2.medianBlur(image_rgb, 5)
    
    # Appliquer la segmentation par Mean Shift
    mean_shift = cv2.pyrMeanShiftFiltering(image_filtered, sp=21, sr=51)
    return(mean_shift)





def kmeans_segmentation(image_path, n_clusters=3):
    # Prétraitement
    image = cv2.imread(image_path)
    
    # Vérifiez si l'image est correctement chargée
    if image is None:
        raise ValueError("L'image n'a pas pu être chargée. Vérifiez le chemin.")
    
    # Assurez-vous que l'image a le bon type
    print("Type d'image:", image.dtype)  # Débogage
    print("Dimensions de l'image:", image.shape)  # Débogage

    # Convertir en RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir en RGB
    image_filtered = cv2.medianBlur(image_rgb, 5)  # Appliquer un flou médian
    flat_image = image_filtered.reshape((-1, 3))  # Aplatir l'image

    # Appliquer K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(flat_image)  # Ajuster le modèle aux pixels de l'image

    # Récupérer les labels et les centres des clusters
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_.astype(int)

    # Reconstruire l'image segmentée
    segmented_image = cluster_centers[labels].reshape(image_filtered.shape)

    # Assurez-vous que l'image segmentée est au format uint8
    segmented_image = segmented_image.astype(np.uint8)

    return segmented_image


def add_noise(image_path, noise_type="gaussian"):
    image=cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Assurez-vous que l'image est dans la plage [0, 1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0  # Normaliser l'image

    # Ajouter le bruit selon le type spécifié
    if noise_type == "gaussian":
        noisy_image = random_noise(image, mode='gaussian', var=0.01)  # Ajouter du bruit gaussien
    elif noise_type == "salt_and_pepper":
        noisy_image = random_noise(image, mode='s&p', salt_vs_pepper=0.5)  # Ajouter du bruit sel et poivre
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    # S'assurer que les valeurs restent dans la plage [0, 1]
    noisy_image = np.clip(noisy_image, 0, 1)

    # Reconversion de l'image en uint8 pour affichage
    noisy_image = (noisy_image * 255).astype(np.uint8)
    return(noisy_image)

def otsu_segmentation(image_path):
    # Convertir l'image en niveaux de gris
    image=cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer le seuillage d'Otsu
    _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (thresh_image)

def threshold_segmentation(image_path, threshold_value=150):
    image=cv2.imread(image_path)
    # Convertir l'image en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Appliquer le seuillage fixe
    _, thresh_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
    return (thresh_image)

def unet(image_path):
    # Fonction pour construire le modèle U-Net
    def build_unet_model(input_shape):
        inputs = Input(input_shape)

        # Partie contractante (encodeur)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        # Partie du "goulot"
        conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

        # Partie expansive (décodeur)
        up6 = UpSampling2D(size=(2, 2))(conv5)
        merge6 = concatenate([up6, conv4], axis=3)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(merge6)
        conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

        up7 = UpSampling2D(size=(2, 2))(conv6)
        merge7 = concatenate([up7, conv3], axis=3)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(merge7)
        conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

        up8 = UpSampling2D(size=(2, 2))(conv7)
        merge8 = concatenate([up8, conv2], axis=3)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(merge8)
        conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

        up9 = UpSampling2D(size=(2, 2))(conv8)
        merge9 = concatenate([up9, conv1], axis=3)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(merge9)
        conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs, outputs)
        return model

    # Charger l'image depuis le chemin et la prétraiter
    img = load_img(image_path, target_size=(256, 256))  # Redimensionner si nécessaire
    img_array = img_to_array(img) / 255.0  # Normaliser l'image
    img_array = np.expand_dims(img_array, axis=0)  # Ajouter une dimension pour le lot

    # Créer et compiler le modèle U-Net
    model = build_unet_model(input_shape=(256, 256, 3))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Prédire la segmentation
    predicted_mask = model.predict(img_array)[0, :, :, 0]  # Récupérer la première image du lot

    # Retourner le masque prédit
    return predicted_mask

