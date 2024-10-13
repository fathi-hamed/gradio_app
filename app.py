import streamlit as st
import os
import numpy as np
from functions import meanshift_segmentation, kmeans_segmentation, add_noise, otsu_segmentation, threshold_segmentation, unet
from PIL import Image
import cv2

# Interface utilisateur
if not os.path.exists("temp"):
    os.makedirs("temp")

def main():
    st.title("Application de Segmentation d'Image 🖼️")

    # Choisir la méthode de segmentation
    method = st.selectbox("Choisissez une méthode de segmentation", 
                          ["MeanShift", "KMeans", "Add Noise", "Seuillage_Otsu", "Seuillage_Threshold"])

    # Télécharger une image
    uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Enregistrer l'image téléchargée dans un dossier temporaire
        image_path = os.path.join("temp", uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Afficher l'image chargée
        st.image(uploaded_file, caption='Image Chargée', use_column_width=True)
        n_clusters = None
        threshold_value = None





        # Appliquer la méthode de segmentation choisie
        if method == "MeanShift":
            st.write("Application de la méthode MeanShift...")
            segmented_image = meanshift_segmentation(image_path)
    
    # Vérifier si c'est un tableau NumPy, aucune conversion RGB n'est nécessaire
            if isinstance(segmented_image, np.ndarray):
        # Affichage directement de l'image segmentée
                    st.image(segmented_image, caption="Image segmentée", use_column_width=True)
        elif method == "KMeans":
            n_clusters = st.slider("Choisissez le nombre de clusters", min_value=2, max_value=10, value=3)
            st.write(f"Nombre de clusters choisi : {n_clusters}")
            st.write("Application de la méthode KMeans...")
            segmented_image = kmeans_segmentation(image_path,n_clusters=n_clusters)
            if isinstance(segmented_image, np.ndarray):
                segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        elif method == "Add Noise":
            st.write("Ajout de bruit à l'image...")
            segmented_image = add_noise(image_path)
            if isinstance(segmented_image, np.ndarray):
                segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
        elif method == "Seuillage_Otsu":
            st.write("Application de la méthode Otsu...")
            segmented_image = otsu_segmentation(image_path)
            # Assurez-vous que l'image est en niveaux de gris
            if len(segmented_image.shape) == 3:
                segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        elif method == "Seuillage_Threshold":
            threshold_value = st.slider("Choisissez une valeur de seuil", min_value=0, max_value=255, value=150)
            st.write(f"Valeur de seuil choisie : {threshold_value}")
            st.write("Application de la méthode Threshold...")
            segmented_image = threshold_segmentation(image_path,threshold_value=threshold_value)
            if len(segmented_image.shape) == 3:
                segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
        

        # Convertir l'image segmentée à un format affichable par Streamlit
        if segmented_image is not None:
            if len(segmented_image.shape) == 2:  # Image en niveaux de gris
                # Convertir l'image numpy en image PIL
                pil_image = Image.fromarray(segmented_image, mode='L')  # mode='L' pour niveaux de gris
            else:  # Image en couleur
                # Convertir l'image numpy en image PIL
                pil_image = Image.fromarray(segmented_image)
                pil_image = pil_image.convert('RGB')  # Assurez-vous que l'image est en RGB

            # Afficher l'image segmentée
            st.image(pil_image, caption='Image Segmentée', use_column_width=True)

if __name__ == "__main__":
    main()
