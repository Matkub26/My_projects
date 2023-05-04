import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import tkinter as tk
from tkinter import filedialog


def load_data(path):
    os.chdir(path)
    images = []
    with os.scandir(path) as files:
        for file in files:
            images.append(file.name)
    return images

def extract_features(file, model):
    with Image.open(file) as img:
        img = img.resize((224, 224))
        img_array = np.asarray(img, dtype=np.float32)

    reshaped_img = img_array.reshape(1, 224, 224, 3)
    imgx = preprocess_input(reshaped_img)
    features = model.predict(imgx, use_multiprocessing=True)
    return features

def preprocess_features(features):
    feat = np.array(list(features.values()))
    feat = feat.reshape(-1, 4096)
    pca = PCA(n_components=20, random_state=22)
    pca.fit(feat)
    x = pca.transform(feat)
    return x

def cluster_images(filenames, features):
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=50).fit_predict(features)
    groups = {}
    for file, cluster in zip(filenames, clustering):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
    return groups

def save_clusters(groups, path):
    for cluster in groups:
        cluster_path = os.path.join(path, f'cluster_{cluster}')
        os.makedirs(cluster_path, exist_ok=True)
        for image in groups[cluster]:
            src_path = os.path.join(path, image)
            dst_path = os.path.join(cluster_path, image)
            os.rename(src_path, dst_path)

def process_images():
    # Load data
    images = load_data(PATH.get())

    # Extract features
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    data = {}
    for image in images:
        feat = extract_features(image, model)
        data[image] = feat

    # Preprocess features
    filenames = np.array(list(data.keys()))
    features = preprocess_features(data)

    # Cluster images
    groups = cluster_images(filenames, features)

    # Save clusters
    save_clusters(groups, PATH.get())

    status_label.config(text='Job is done')


if __name__ == '__main__':
    root = tk.Tk()
    root.title("Image Clustering")

    label_title = tk.Label(root, text="Image Clustering Application", font=("Arial", 20))
    label_title.pack(pady=20)

    path_label = tk.Label(root, text="Enter folder path:")
    path_label.pack()

    PATH = tk.StringVar()
    path_entry = tk.Entry(root, textvariable=PATH, width=50)
    path_entry.pack(pady=5)

    browse_button = tk.Button(root, text="Browse", command=lambda: PATH.set(filedialog.askdirectory()))
    browse_button.pack(pady=5)

    run_button = tk.Button(root, text="Run Script", command=process_images)
    run_button.pack(pady=10)

    status_label = tk.Label(root, text="")
    status_label.pack(pady=10)

    root.mainloop()
