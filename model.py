import numpy as np
import matplotlib.pyplot as plt
import glob
import cv2
import tensorflow as tf
#from tensorflow.keras.layers import BatchNormalization
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import os
import seaborn as sns
from sklearn import preprocessing
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

print(os.listdir("./archive"))

SIZE = 64
train_images = []
train_labels = []
for directory_path in glob.glob("./archive/BananaLSD/OriginalSet/*"):
    label = directory_path.split("\\")[-1]
    print(label)
    for img_path in glob.glob(os.path.join(directory_path, "*.jpeg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        train_images.append(img)
        train_labels.append(label)
train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images = []
test_labels = []
for directory_path in glob.glob("./archive/BananaLSD/AugmentedSet/*"):
    fruit_label = directory_path.split("\\")[-1]
    for img_path in glob.glob(os.path.join(directory_path, "*.jpeg")):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (SIZE, SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        test_images.append(img)
        test_labels.append(fruit_label)
test_images = np.array(test_images)
test_labels = np.array(test_labels)

le = preprocessing.LabelEncoder()
le.fit(test_labels)
test_labels_encoded = le.transform(test_labels)
le.fit(train_labels)
train_labels_encoded = le.transform(train_labels)

x_train, y_train, x_test, y_test = train_images, train_labels_encoded, test_images, test_labels_encoded

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)
activation = 'sigmoid'
feature_extractor = Sequential()
feature_extractor.add(Conv2D(32, 3, activation=activation, input_shape=(SIZE, SIZE, 3)))
feature_extractor.add(MaxPooling2D(pool_size=(2, 2), strides=2))
feature_extractor.add(Conv2D(32, 3, activation=activation, input_shape=(SIZE, SIZE, 3)))
feature_extractor.add(MaxPooling2D(pool_size=(2, 2), strides=2))

feature_extractor.add(Flatten())
x = feature_extractor.output
x = Dense(128, activation=activation)(x)
prediction_layer = Dense(2, activation='softmax')(x)

cnn_model = Model(inputs=feature_extractor.input, outputs=prediction_layer)
cnn_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print(cnn_model.summary())

history = cnn_model.fit(x_train, y_train_one_hot, epochs=10, validation_data=(x_test, y_test_one_hot))

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

prediction_NN = cnn_model.predict(x_test)
prediction_NN = np.argmax(prediction_NN, axis=-1)
prediction_NN = le.inverse_transform(prediction_NN)

cm = confusion_matrix(tuple(test_labels.tolist()), tuple(prediction_NN.tolist()))
print(cm)
sns.heatmap(cm, annot=True)

n = 9
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0)
prediction = np.argmax(cnn_model.predict(input_img))
prediction = le.inverse_transform([prediction])
print("The prediction for this image is: ", prediction)
print("The actual label for this image is: ", test_labels[n])

X_for_RF = feature_extractor.predict(x_train)

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_for_RF, y_train)

X_test_feature = feature_extractor.predict(x_test)

prediction_RF = classifier.predict(X_test_feature)
prediction_RF = le.inverse_transform(prediction_RF)

print("Accuracy = ", metrics.accuracy_score(test_labels, prediction_RF))

cm = confusion_matrix(tuple(test_labels.tolist()), tuple(prediction_RF.tolist()))
sns.heatmap(cm, annot=True)

n = 9
img = x_test[n]
plt.imshow(img)
input_img = np.expand_dims(img, axis=0)
input_img_features = feature_extractor.predict(input_img)
prediction_RF = classifier.predict(input_img_features)[0]
prediction_RF = le.inverse_transform([prediction_RF])
print("The prediction for this image is: ", prediction_RF)
print("The actual label for this image is: ", test_labels[n])
