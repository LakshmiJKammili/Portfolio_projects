import pickle

# Load the pickle file
with open("ref_name.pkl", "rb") as f:
    data = pickle.load(f)

# Inspect the loaded object
print(data)

