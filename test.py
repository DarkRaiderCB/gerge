import pickle

with open('polyvore_features.pkl', 'rb') as f:
    data = pickle.load(f)

print(data['image_paths'][:5])  # Print first 5 image paths to verify
