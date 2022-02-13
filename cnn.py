import os

import seaborn
import seaborn as sns
import sns
from keras.activations import relu, softmax
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

print(tf.__version__)
import scikeras as SciKeras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from scikeras.wrappers import KerasClassifier, KerasRegressor
import numpy
import tensorflow
import tf as tf
from keras import Sequential
from keras.layers import Reshape, Conv2D, Activation, Dropout, MaxPooling2D, Flatten, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from numpy import array
from skimage import io
from sklearn.model_selection import train_test_split
from tensorflow import keras

from main import *

y_labels_new_num = []
objects = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
for i in range(len(y_labels)):
#for i in range(len(y_labels_new)):
    if y_labels[i] == 'akiec':
        y_labels_new_num.append(0)
    elif y_labels[i] == 'bcc':
        y_labels_new_num.append(1)
    elif y_labels[i] == 'bkl':
        y_labels_new_num.append(2)
    elif y_labels[i] == 'df':
        y_labels_new_num.append(3)
    elif y_labels[i] == 'mel':
        y_labels_new_num.append(4)
    elif y_labels[i] == 'nv':
        y_labels_new_num.append(5)
    elif y_labels[i] == 'vasc':
        y_labels_new_num.append(6)


X_train, X_test, y_train, y_test = train_test_split(X_images, y_labels_new_num, test_size=0.2, random_state=101)


def CNNClassifier(epochs=30, batch_size=10, layers=3, dropout=0.5, activation='relu'):
    def set_params():
        i = 1

    def create_model():
        model = Sequential()
        model.add(Reshape((IMG_WIDTH, IMG_HEIGHT, 3)))

        for i in range(layers):
            model.add(Conv2D(64, (3, 3), padding='same'))
            model.add(Activation(activation))
            model.add(Dropout(dropout / 2.0))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout / 2.0))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation(activation))
        model.add(Conv2D(128, (3, 3)))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout / 2.0))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation(activation))
        model.add(Dropout(dropout))
        model.add(Dense(7))
        model.add(Activation('softmax'))

        # initiate RMSprop optimizer
        opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

        # train the model using RMSprop
        model.compile(loss='categorical_crossentropy',
                      optimizer=opt,
                      metrics=[tensorflow.keras.metrics.AUC(), 'accuracy'])
        return model

    return KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=1)


X_train = numpy.array(X_train)
X_test = numpy.array(X_test)
y_train = numpy.array(y_train)
y_test = numpy.array(y_test)

y_train_onehot = np.zeros((y_train.size, y_train.max().astype(int) + 1))
y_train_onehot[np.arange(y_train.size), y_train.astype(int)] = 1

y_test_onehot = np.zeros((y_test.size, y_test.max().astype(int) + 1))
y_test_onehot[np.arange(y_test.size), y_test.astype(int)] = 1

cnn = CNNClassifier()

#cnn.fit(X_train.astype(np.float32), y_train_onehot.astype(np.float32),
  #      validation_data=(X_test.astype(np.float32), y_test_onehot.astype(np.float32))
   #     , verbose=1)


# @title Definition for the model_stats() function { display-mode: "form" }
def model_stats(name, y_test, y_pred, y_pred_proba):
    cm = confusion_matrix(y_test, y_pred)

    print(name)

    accuracy = accuracy_score(y_test, y_pred)
    print("The accuracy of the model is " + str(round(accuracy, 5)))

    y_test_onehot = np.zeros((y_test.size, y_test.max().astype(int) + 1))
    y_test_onehot[np.arange(y_test.size), y_test.astype(int)] = 1

    roc_score = roc_auc_score(y_test_onehot, y_pred_proba)

    print("The ROC AUC Score of the model is " + str(round(roc_score, 5)))

    return cm


def plot_cm(name, cm):
    classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    df_cm = pd.DataFrame(cm, index=[i for i in classes], columns=[i for i in classes])
    df_cm = df_cm.round(5)

    plt.figure(figsize=(12, 8))
    seaborn.heatmap(df_cm, annot=True, fmt='g')
    plt.title(name + " Model Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


#y_pred = cnn.predict(X_test)
#y_pred_proba = cnn.predict_proba(X_test)
#cnn_cm = model_stats("CNN", y_test, y_pred, y_pred_proba)

#plot_cm("CNN", cnn_cm)
