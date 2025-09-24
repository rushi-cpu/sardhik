# train_model.py
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)

# Save to pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
