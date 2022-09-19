# Pandas is used for data manipulation
import pandas as pd
# Use numpy to convert to arrays
import numpy as np
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Import the random forests model
from sklearn.ensemble import RandomForestRegressor
import sys
from PIL import Image
from matplotlib import pyplot as plt
import cv2

# Read in data
features = pd.read_csv("/work/shared/DEVCOM-SC21/Network/data/training_5000pix_200m.csv")

# Labels are the values we want to predict (right now I'm just trying to remove z)
xlabels = np.array(features['lat_dif'])
ylabels = np.array(features['lon_dif'])
zlabels = np.array(features['alt_dif'])
# Remove the labels from the features
# axis 1 refers to the columns
features= features.drop('lat_dif', axis = 1)
features= features.drop('lon_dif', axis = 1)
features= features.drop('alt_dif', axis = 1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)

#print([features[0,6]])
img = cv2.imread((features[0,6]), 0)
#print(img)
img = cv2.resize(img, (240,426))
#img = img.convert('1') #turns it black and white
image_array = np.array(img, dtype='uint8')
image_array = np.reshape(image_array, (1,-1))
#print(image_array)
print("image size:", image_array.shape)

#turn images into black and white arrays
for i in range(len(features)-1):
  img = cv2.imread((features[i+1,6]), 0)
  #img = img.convert('1') #turns it black and white
  img = cv2.resize(img,(240,426))
  image_vector =  np.reshape(np.array(img, dtype='uint8'), (1,-1))
  image_array = np.concatenate((image_array, image_vector), axis=0)
  print(i)
#print(image_array)
  #features[i,1] = np.array(img, dtype='uint8')
print("init image_array size:", image_array.size)

img1 = cv2.imread((features[0,-1]), 0)
#print(img)
img1 = cv2.resize(img1, (240,426))
#img = img.convert('1') #turns it black and white
image_array1 = np.array(img1, dtype='uint8')
image_array1 = np.reshape(image_array1, (1,-1))
#print(image_array)
print("image size:", image_array1.shape)

#turn images into black and white arrays
for i in range(len(features)-1):
  img1 = cv2.imread((features[i+1,-1]), 0)
  #img = img.convert('1') #turns it black and white
  img1 = cv2.resize(img1,(240,426))
  image_vector1 =  np.reshape(np.array(img1, dtype='uint8'), (1,-1))
  image_array1 = np.concatenate((image_array1, image_vector1), axis=0)
  print(i)
#print(image_array)
  #features[i,1] = np.array(img, dtype='uint8')
print("2nd image_array size:", image_array1.size)

#concat both vectors:
image_array=np.concatenate((image_array, image_array1), axis=1)
print("image_array size:", image_array.size)

# Split the data into training and testing sets
train_features, test_features, train_xlabels, test_xlabels, train_ylabels, test_ylabels, train_zlabels, test_zlabels,= train_test_split(image_array, xlabels, ylabels, zlabels, test_size = 0.2, random_state = 42)

# Instantiate model with 100 decision trees
rf_z = RandomForestRegressor(n_estimators = 100, random_state = 42)
# Train the model on training data
rf_z.fit(train_features, train_zlabels);

# Use the forest's predict method on the test data
predictions = rf_z.predict(test_features)
# Calculate the absolute errors
z_errors = (predictions - test_zlabels)**2
# Print out the mean absolute error (mae)
print('Mean Absolute Error for z:', round(np.mean(z_errors), 10), 'meters.')

# Instantiate model with 100 decision trees
rf_x = RandomForestRegressor(n_estimators = 100, random_state = 42)
# Train the model on training data
rf_x.fit(train_features, train_xlabels);

# Use the forest's predict method on the test data
predictions = rf_x.predict(test_features)
# Calculate the absolute errors
x_errors = (predictions - test_xlabels)**2
# Print out the mean absolute error (mae)
print('Mean Absolute Error for x:', round(np.mean(x_errors), 10), 'degrees.')

# Instantiate model with 100 decision trees
rf_y = RandomForestRegressor(n_estimators = 100, random_state = 42)
# Train the model on training data
rf_y.fit(train_features, train_ylabels);

# Use the forest's predict method on the test data
predictions = rf_y.predict(test_features)
# Calculate the absolute errors
y_errors = (predictions - test_ylabels)**2
# Print out the mean absolute error (mae)
print('Mean Absolute Error for y:', round(np.mean(y_errors), 10), 'degrees.')
