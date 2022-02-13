import random

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy
from matplotlib import pyplot as plt
import cv2
import skimage.io as io
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
from cv2 import imshow
print(cv2.__version__)
import pandas as pd
from tqdm import tqdm
import cv2
import csv
import os
import np

import numpy as np
import pandas as pd
import cv2 as cv
from PIL import Image
import matplotlib.pylab as plt
from numpy import array, asarray
from sys import getsizeof

# load and show an image with Pillow
from PIL import Image, ImageOps

from main import *


X_gray_train, X_gray_test, y_train, y_test = train_test_split(X_images_new, y_labels_new, test_size=0.2, random_state=101)


knn = KNeighborsClassifier(n_neighbors=30)

X_gray_train_flat = []
X_gray_test_flat = []


for x in range(len(X_gray_train)):
    image = io.imread('/Users/janvisrivastava/PycharmProjects/SkinCancerKNN/images/' + X_gray_train[x] + '.jpg')
    array1 = array(image)
    array2 = array1.flatten()
    X_gray_train_flat.append(array2)

for x in range(len(X_gray_test)):
    image = io.imread('/Users/janvisrivastava/PycharmProjects/SkinCancerKNN/images/' + X_gray_test[x] + '.jpg')
    array1 = array(image)
    array2 = array1.flatten()
    X_gray_test_flat.append(array2)

knn.fit(X_gray_train_flat, y_train)


def model_stats(name, y_test, y_pred, y_pred_proba):
    cm = confusion_matrix(y_test, y_pred)

    print(name)

    accuracy = accuracy_score(y_test, y_pred)
    print("The accuracy of the model is " + str(round(accuracy, 5)))

    roc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')

    print("The ROC AUC Score of the model is " + str(round(roc_score, 5)))

    return cm

y_pred = knn.predict(X_gray_test_flat)
y_pred_proba = knn.predict_proba(X_gray_test_flat)

knn_cm = model_stats("K Nearest Neighbors",y_test,y_pred,y_pred_proba)


def plot_cm(name, cm):
  classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

  df_cm = pd.DataFrame(cm, index = [i for i in classes], columns = [i for i in classes])
  df_cm = df_cm.round(5)

  plt.figure(figsize = (12,8))
  sns.heatmap(df_cm, annot=True, fmt='g')
  plt.title(name + " Model Confusion Matrix")
  plt.xlabel("Predicted Label")
  plt.ylabel("True Label")
  plt.show()

plot_cm("K Nearest Neighbors",knn_cm)