import os
import cv2
import joblib
import numpy as np
import tensorflow as tf
from scipy.signal import wiener
from skimage.feature import hog, local_binary_pattern
import matplotlib.pyplot as plt

img_size = (128, 128)
seg_img_size = 256

def preprocess_image(img):

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to double [0,1]
    img_gray = img_gray.astype(np.float64) / 255.0

    # Median filtering
    img_med = cv2.medianBlur((img_gray * 255).astype(np.uint8), 3)
    img_med = img_med.astype(np.float64) / 255.0

    # FFT periodic noise removal
    F = np.fft.fftshift(np.fft.fft2(img_med))
    M, N = img_med.shape

    H = np.ones((M, N), dtype=np.float64)

    X, Y = np.meshgrid(np.arange(1, N + 1), np.arange(1, M + 1))

    cx = N // 2 + 1
    cy = M // 2 + 1

    spikes = [
        (-10, -6),
        (10, -6)
    ]

    DR = 5

    for ux, uy in spikes:

        D0 = np.sqrt((X - (cx + ux))**2 + (Y - (cy + uy))**2)
        D1 = np.sqrt((X - (cx - ux))**2 + (Y - (cy - uy))**2)

        H[D0 <= DR] = 0
        H[D1 <= DR] = 0

    F_filtered = F * H

    restored = np.real(np.fft.ifft2(np.fft.ifftshift(F_filtered)))

    # Normalize
    restored = restored - restored.min()

    if restored.max() > 0:
        restored = restored / restored.max()

    # Wiener filtering
    restored = wiener(restored, (3, 3))

    restored = np.nan_to_num(restored)
    restored = np.clip(restored, 0, 1)

    # Convert to uint8 grayscale
    restored_uint8 = (restored * 255).astype(np.uint8)

    return restored_uint8



def extract_hog_lbp_orb_features(img_path):

    img = cv2.imread(img_path)

    if img is None:
        raise Exception("Image not found!")

    # Preprocessing
    gray = preprocess_image(img)
    gray = cv2.resize(gray, img_size)

    # HOG
    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True
    )

    # LBP
    radius = 1
    n_points = 8 * radius

    lbp = local_binary_pattern(
        gray,
        n_points,
        radius,
        method="uniform"
    )

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2)
    )

    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    # ORB
    orb = cv2.ORB_create(nfeatures=500)

    keypoints, descriptors = orb.detectAndCompute(gray, None)

    max_orb_features = 500 * 32

    if descriptors is not None:
        orb_features = descriptors.flatten()

        if len(orb_features) < max_orb_features:
            orb_features = np.pad(
                orb_features,
                (0, max_orb_features - len(orb_features))
            )
        else:
            orb_features = orb_features[:max_orb_features]

    else:
        orb_features = np.zeros(max_orb_features)

    orb_features = orb_features.astype(np.float32)
    orb_features /= 255.0

    features = np.hstack([
        hog_features,
        lbp_hist,
        orb_features
    ])

    return features


def refine_mask(pred_mask, threshold=0.25):

    binary = (pred_mask > threshold).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3)
    )

    cleaned = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )

    return cleaned



def run_segmentation(image_path, seg_model, threshold=0.25, refine=True):

    img = cv2.imread(image_path)

    if img is None:
        raise Exception("Image not found!")

    original_h, original_w = img.shape[:2]

    img = preprocess_image(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_resized = cv2.resize(
        img_rgb,
        (seg_img_size, seg_img_size),
        interpolation=cv2.INTER_LINEAR
    )

    img_input = img_resized.astype(np.float32)

    img_input = tf.keras.applications.mobilenet_v2.preprocess_input(img_input)

    img_input = np.expand_dims(img_input, axis=0)

    pred_mask = seg_model.predict(img_input, verbose=0)[0, :, :, 0]

    if refine:
        mask = refine_mask(pred_mask, threshold=threshold)
    else:
        mask = (pred_mask > threshold).astype(np.uint8)

    mask_original = cv2.resize(
        mask,
        (original_w, original_h),
        interpolation=cv2.INTER_NEAREST
    )

    mask_original = (mask_original > 0).astype(np.uint8)*255

    return mask_original



def predict(dataset_path):

    import matplotlib.pyplot as plt

    # Load trained models
    model = joblib.load("bestest.pkl")

    seg_model = tf.keras.models.load_model(
        "best_unet_mobilenetv2.h5",
        compile=False
    )

    # Get all image paths from dataset folder
    image_paths = []

    valid_extensions = [".png", ".jpg", ".jpeg", ".bmp"]

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                image_paths.append(os.path.join(root, file))

    image_paths = sorted(image_paths)


    predictions = []

    for image_path in image_paths:

        features = extract_hog_lbp_orb_features(image_path)

        prediction = model.predict([features])[0]
        prediction = prediction.lower()

        predictions.append(prediction)

    # Show the classification output first
    print("Class predictions for all samples:")
    print(predictions)


    for image_path, prediction in zip(image_paths, predictions):

        if prediction == "safe":
            print(os.path.basename(image_path), "-> safe, no segmentation")
            continue

        # Run segmentation only if not safe
        mask = run_segmentation(
            image_path,
            seg_model,
            threshold=0.25,
            refine=True
        )

        # Read original image for display
        img = cv2.imread(image_path)

        if img is None:
            raise Exception("Image not found!")

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        print(os.path.basename(image_path), "->", prediction)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title("Input image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap="gray", vmin=0, vmax=255)
        plt.title("Mask for class: " + prediction)
        plt.axis("off")

        plt.show()

    return predictions
