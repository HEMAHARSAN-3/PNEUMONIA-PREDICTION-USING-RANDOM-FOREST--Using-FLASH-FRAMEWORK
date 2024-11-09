# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function to load and preprocess images
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = image / 255.0  # Normalize to [0, 1] range
    return image

# Step 1: Load the dataset from CSV
csv_file_path = r"E:\pneumonia.csv"  # Specify your CSV file path here
df = pd.read_csv(csv_file_path)

# Ensure the paths in the CSV are correct relative to the script location
# If necessary, adjust the paths by adding the base directory
base_dir = ''  # Set this to the base directory if the paths in CSV are relative
df['image_path'] = df['image_path'].apply(lambda x: os.path.join(base_dir, x))

# Step 2: Apply the function to the image paths
df['image'] = df['image_path'].apply(load_and_preprocess_image)

# Convert images to numpy array
X = np.stack(df['image'].values)
y = df['label'].values

# Flatten images for Random Forest (if images are used directly)
X = X.reshape(X.shape[0], -1)

# Encode labels to numerical values
y = np.where(y == 'normal', 0, 1)  # 0 for normal, 1 for pneumonia

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Step 6: Save the model
joblib.dump(rf, 'random_forest_model.pkl')

# Indicate that the process is complete
print("Model training, evaluation, and saving complete.")
