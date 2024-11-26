import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from glob import glob
import os
import matplotlib.pyplot as plt

image_paths = glob(data\*\*.jpg')   
print("Number of images found:", len(image_paths))  

def extract_harris_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    keypoints = np.argwhere(corners > 0.01 * corners.max())
    return keypoints


all_features = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        continue  
    keypoints = extract_harris_features(image)
    all_features.append(keypoints)


num_clusters = 100
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(np.vstack(all_features))

def create_bow_histogram(image, kmeans):
    keypoints = extract_harris_features(image)
    if keypoints.size == 0:
        return np.zeros(num_clusters)
    labels = kmeans.predict(keypoints)
    histogram, _ = np.histogram(labels, bins=np.arange(num_clusters + 1))
    return histogram

X = []
y = []
for image_path in image_paths:
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        continue  
    bow_hist = create_bow_histogram(image, kmeans)
    X.append(bow_hist)
    
   
    label = os.path.basename(os.path.dirname(image_path))
    y.append(label)

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

class BoundingBox:
    def __init__(self, min_x, min_y, max_x, max_y, label):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.label = label
    
    def draw_on_image(self, image):
        
        cv2.rectangle(image, (self.min_y, self.min_x), (self.max_y, self.max_x), (0, 255, 0), 2)
        
        cv2.putText(image, self.label, (self.min_y, self.min_x - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

test_image_paths = glob(r'C:\Users\Hp\Desktop\UNIVERSITY\CV\task3-immplimant\test\*.jpg')  
print("Number of test images found:", len(test_image_paths))

for test_image_path in test_image_paths:
    test_image = cv2.imread(test_image_path)
    if test_image is None:
        print(f"Error loading test image: {test_image_path}")
        continue  
    
    keypoints = extract_harris_features(test_image)

    if keypoints.size == 0:
        print("No keypoints found.")
    else:
        min_x, min_y = np.min(keypoints, axis=0)
        max_x, max_y = np.max(keypoints, axis=0)
        
        label = svm.predict([create_bow_histogram(test_image, kmeans)])[0]  
        bounding_box = BoundingBox(min_x, min_y, max_x, max_y, label)
        
        bounding_box_image = test_image.copy()
        bounding_box.draw_on_image(bounding_box_image)
        
        bounding_box_image_rgb = cv2.cvtColor(bounding_box_image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(bounding_box_image_rgb)
        plt.axis('off')
        plt.title(f"Test Image with Bounding Box and Label: {label}")
        plt.show()
