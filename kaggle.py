import kagglehub

# This line downloads the dataset (only the first time) and returns the local path
path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")

print("Path to dataset files:", path)
