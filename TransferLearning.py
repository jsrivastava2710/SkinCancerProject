import tf
from keras import Sequential
from keras.layers import Dropout, BatchNormalization, Dense
from scikeras.wrappers import KerasClassifier
from tensorflow import keras
from tensorflow.python.keras.applications.mobilenet import MobileNet
import tensorflow as tf
from main import *
from cnn import *


def transfer_learning_model():
    mobilenet_model = MobileNet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, pooling="max")

    transfer_model = Sequential()
    transfer_model.add(mobilenet_model)
    transfer_model.add(Dropout(0.1))
    transfer_model.add(BatchNormalization())
    transfer_model.add(Dense(256, activation="relu"))
    transfer_model.add(Dropout(0.1))
    transfer_model.add(BatchNormalization())
    transfer_model.add(Dense(7, activation="softmax"))

    # initiate RMSprop optimizer
    opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    transfer_model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=[tensorflow.keras.metrics.AUC()])

    return transfer_model


print(X_train.shape)
print(y_train_onehot.shape)

print(X_test.shape)
print(y_test_onehot.shape)

transfer_model = KerasClassifier(build_fn=transfer_learning_model, verbose=1, epochs=20)
# @title Instructor Solution { display-mode: "form" }
transfer_model.fit(X_train.astype(np.float32), y_train_onehot.astype(np.float32),
        validation_data=(X_test.astype(np.float32), y_test_onehot.astype(np.float32))
        , verbose=1)

#@title Instructor Solution { display-mode: "form" }
y_pred = transfer_model.predict(X_test)
y_pred_proba = transfer_model.predict_proba(X_test)
transfer_cm = model_stats("Transfer CNN",y_test,y_pred,y_pred_proba)

plot_cm("Transfer Learning CNN", transfer_cm)