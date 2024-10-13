# import gradio as gr
# import cv2
# import numpy as np
# from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans

# # MeanShift segmentation function
# def meanshift_segmentation(image):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_filtered = cv2.medianBlur(image_rgb, 5)
#     flat_image = image_filtered.reshape((-1, 3))
    
#     bandwidth = estimate_bandwidth(flat_image, quantile=0.2, n_samples=500)
#     ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
#     ms.fit(flat_image)
    
#     labels = ms.labels_
#     cluster_centers = ms.cluster_centers_
#     segmented_image = cluster_centers[labels].reshape(image_filtered.shape).astype(np.uint8)
    
#     return segmented_image

# # KMeans segmentation function
# def kmeans_segmentation(image, n_clusters=2):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_filtered = cv2.medianBlur(image_rgb, 5)
#     flat_image = image_filtered.reshape((-1, 3))
    
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     kmeans.fit(flat_image)
    
#     labels = kmeans.labels_
#     cluster_centers = kmeans.cluster_centers_
#     segmented_image = cluster_centers[labels].reshape(image_filtered.shape).astype(np.uint8)
    
#     return segmented_image

# # Wrapper function for the Gradio interface
# def segment_image(image, method):
#     if method == "MeanShift":
#         return meanshift_segmentation(image)
#     elif method == "KMeans":
#         return kmeans_segmentation(image)

# # Gradio interface
# interface = gr.Interface(
#     fn=segment_image, 
#     inputs=[
#         gr.Image(type="numpy", label="Upload an Image"),  # Image input
#         gr.Radio(choices=["MeanShift", "KMeans"], label="Segmentation Method")  # Segmentation method selection
#     ],
#     outputs=gr.Image(type="numpy", label="Segmented Image"),  # Output segmented image
#     title="Image Segmentation App",
#     description="Upload an image and choose a segmentation method to segment the image."
# )

# # Launch the app
# interface.launch(share=True)



import gradio as gr
import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr

# MeanShift segmentation function
def meanshift_segmentation(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_filtered = cv2.medianBlur(image_rgb, 5)
    flat_image = image_filtered.reshape((-1, 3))
    
    bandwidth = estimate_bandwidth(flat_image, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(flat_image)
    
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    segmented_image = cluster_centers[labels].reshape(image_filtered.shape).astype(np.uint8)
    
    return segmented_image

# KMeans segmentation function
def kmeans_segmentation(image, n_clusters=2):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_filtered = cv2.medianBlur(image_rgb, 5)
    flat_image = image_filtered.reshape((-1, 3))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(flat_image)
    
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    segmented_image = cluster_centers[labels].reshape(image_filtered.shape).astype(np.uint8)
    
    return segmented_image

# Denoising functions
def add_noise(image, noise_type="gaussian"):
    if noise_type == "gaussian":
        noisy_image = random_noise(image, mode='gaussian', var=0.01)
    elif noise_type == "salt_and_pepper":
        noisy_image = random_noise(image, mode='s&p', salt_vs_pepper=0.5)
    return np.clip(noisy_image, 0, 1)

def apply_filters(image, noise_type="gaussian"):
    noisy_image = add_noise(image, noise_type)
    
    noisy_image_float = noisy_image.astype(np.float32)

    avg_filtered = cv2.blur(noisy_image_float, (5, 5))
    gaussian_filtered = cv2.GaussianBlur(noisy_image_float, (5, 5), 0)
    sigma_est = np.mean(estimate_sigma(noisy_image))
    nl_means_filtered = denoise_nl_means(noisy_image, h=1.15 * sigma_est, fast_mode=True, patch_size=5, patch_distance=6)

    return noisy_image, avg_filtered, gaussian_filtered, nl_means_filtered

# Wrapper function for segmentation and denoising
def segment_and_denoise(image, method, noise_type):
    segmented_image = None
    if method == "MeanShift":
        segmented_image = meanshift_segmentation(image)
    elif method == "KMeans":
        segmented_image = kmeans_segmentation(image)

    # Denoising
    noisy_image, avg_filtered, gaussian_filtered, nl_means_filtered = apply_filters(image, noise_type)

    return segmented_image, noisy_image, avg_filtered, gaussian_filtered, nl_means_filtered

# Gradio interface
interface = gr.Interface(
    fn=segment_and_denoise,
    inputs=[
        gr.Image(type="numpy", label="Upload an Image"),  # Image input
        gr.Radio(choices=["MeanShift", "KMeans"], label="Segmentation Method"),  # Segmentation method selection
        gr.Radio(choices=["gaussian", "salt_and_pepper"], label="Noise Type")  # Noise type selection
    ],
    outputs=[
        gr.Image(type="numpy", label="Segmented Image"),  # Output segmented image
        gr.Image(type="numpy", label="Noisy Image"),  # Output noisy image
        gr.Image(type="numpy", label="Averaging Filter"),  # Output averaging filter image
        gr.Image(type="numpy", label="Gaussian Filter"),  # Output Gaussian filter image
        gr.Image(type="numpy", label="NLMeans Filter")  # Output NLMeans filter image
    ],
    title="Image Segmentation and Denoising App",
    description="Upload an image, choose a segmentation method and a noise type to apply filters and segment the image."
)

# Launch the app
interface.launch(share=True)
