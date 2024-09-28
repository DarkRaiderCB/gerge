import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import Inception_V3_Weights
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

# Streamlit page configuration
st.set_page_config(page_title="Polyvore Outfit Recommender", layout="wide")

# Title
st.title("👗 Polyvore Outfit Recommender")

# Sidebar for user input
st.sidebar.header("Upload Your Outfit")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

# Transformation for InceptionV3
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load precomputed features


@st.cache_resource
def load_features(feature_file='polyvore_features.pkl'):
    if not os.path.exists(feature_file):
        st.error(f"Feature file '{feature_file}' not found.")
        st.stop()
    with open(feature_file, 'rb') as f:
        data = pickle.load(f)
    all_features = data['features']
    all_labels = data['labels']
    all_image_paths = data['image_paths']
    classes = data['classes']
    return all_features, all_labels, all_image_paths, classes


all_features, all_labels, all_image_paths, classes = load_features()

# Display available categories in the sidebar
if classes:
    st.sidebar.subheader("Available Categories")
    st.sidebar.write(", ".join(classes))
else:
    st.sidebar.warning("No categories available.")

# Normalize features for cosine similarity


@st.cache_resource
def normalize_features(_all_features):
    all_features_np = _all_features.numpy()
    normalized_features_np = normalize(all_features_np, axis=1)
    return torch.tensor(normalized_features_np)  # Keep on CPU for simplicity


normalized_features = normalize_features(all_features)

# Load the pre-trained InceptionV3 model


@st.cache_resource
def load_model():
    weights = Inception_V3_Weights.DEFAULT
    inception = models.inception_v3(
        weights=weights, aux_logits=True)  # Enable aux_logits
    inception.fc = nn.Identity()  # Replace the final layer with Identity
    # Disable auxiliary logits by replacing with Identity
    inception.AuxLogits = nn.Identity()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception.to(device)
    inception.eval()
    return inception, device


inception, device = load_model()

# Function to extract features from an image


def extract_feature(image, model, transform, device):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(image)
        feature = F.normalize(feature, p=2, dim=1)  # Normalize
    return feature.cpu()

# Function to find compatible items


def find_compatible_items(query_feature, all_features, all_labels, all_image_paths, target_categories, top_k=5):
    compatibility = {}
    target_indices = [classes.index(cat)
                      for cat in target_categories if cat in classes]

    if not target_indices:
        st.warning(
            "No valid target categories found. Please check the category names and their cases.")
        return compatibility

    # Compute cosine similarity (since features are normalized, cosine similarity is the dot product)
    similarities = torch.matmul(query_feature, all_features.T).squeeze(
        0)  # Shape: (num_images,)

    for cat_idx in target_indices:
        cat_item_indices = [i for i, label in enumerate(
            all_labels) if label == cat_idx]
        if not cat_item_indices:
            continue
        cat_similarities = similarities[cat_item_indices]
        topk = min(top_k, len(cat_similarities))
        topk_values, topk_indices = torch.topk(cat_similarities, topk)
        topk_original_indices = [cat_item_indices[i] for i in topk_indices]
        compatibility[classes[cat_idx]] = [all_image_paths[i]
                                           for i in topk_original_indices]
    return compatibility

# Function to display compatibility


def display_compatibility(compatible_items, image_width=150):
    """
    Displays the recommended items.

    Parameters:
    - compatible_items (dict): Dictionary with categories as keys and list of image paths as values.
    - image_width (int): Width of the displayed images in pixels.
    """
    num_categories = len(compatible_items)
    num_items = max(len(items)
                    for items in compatible_items.values()) if compatible_items else 0

    if num_categories == 0 or num_items == 0:
        st.info("No compatible items found for the selected categories.")
        return

    # Display compatible items
    for category, images in compatible_items.items():
        st.subheader(f"✨ Top {len(images)} Recommendations for **{category}**")
        cols = st.columns(len(images))
        for idx, img_path in enumerate(images):
            with cols[idx]:
                try:
                    img = Image.open(img_path).convert('RGB')
                    st.image(img, width=image_width)
                except Exception as e:
                    st.write("❌ Image not found or cannot be opened.")


# When the user uploads an image
if uploaded_file is not None:
    try:
        input_image = Image.open(uploaded_file).convert('RGB')

        # Display the uploaded image in the sidebar with reduced size
        st.sidebar.subheader("🖼️ Uploaded Image")
        st.sidebar.image(input_image, width=200)  # Adjust the width as needed

        with st.spinner('🔍 Processing...'):
            # Extract feature
            query_feature = extract_feature(
                input_image, inception, transform, device)

            # Allow user to select target categories
            target_categories = st.multiselect(
                "🎯 Select target categories for recommendations:",
                options=classes,
                default=['pants', 'shoes'],
                help="Choose one or more categories to get compatible items."
            )

            # Find compatible items
            compatible_items = find_compatible_items(
                query_feature, normalized_features, all_labels, all_image_paths, target_categories, top_k=5)

        # Display recommendations
        # Set desired width
        display_compatibility(compatible_items, image_width=150)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
else:
    st.info("🕒 Please upload an image to get recommendations.")
