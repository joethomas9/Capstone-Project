#!/usr/bin/env python3
import os, numpy as np, cv2
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


training_dir = ''
global cnn_model, labels


# -------------------------------------------------------------------------------
# Function to build the training set
# -------------------------------------------------------------------------------

def data_augment(direct):
    # https://github.com/bnsreenu/python_for_microscopists/blob/master/127_data_augmentation_using_keras.py
    # code is from the link above
    # this function traverses all the directories containing images used to train
    # the model, and expands on the collection of images available to train the model
    # by performing data augmentation. This should increase accuracy of the model in
    # the classification of different objects, as well as stopping over-fitting. The
    # generated files are appended to the existing directories.

    datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='constant', cval=125
    )

    i = 0
    print(training_dir+"/"+direct)
    for batch in datagen.flow_from_directory(directory=training_dir+"/"+direct+"/",
                                             batch_size=16,
                                             target_size=(64, 64),
                                             color_mode='rgb',
                                             save_to_dir=training_dir+"/"+direct+"/"+direct,
                                             save_prefix='aug_',
                                             save_format='png'):
        i += 1
        if i > 31:
            break


def set_build(augment):
    os.system('find . -name ".DS_Store" -delete')
    global labels
    # https://github.com/codebasics/deep-learning-keras-tf-tutorial/blob/master/17_data_augmentation/cnn_flower_image_classification_data_augmentations.ipynb
    # code influenced from link above
    # This function reads in the images from the given training directories, and
    # appends them to a dictionary of all related images. A second dictionary is
    # also generated simultaneously, storing the label of the image class.

    # this function builds the training sets for the CNN, classifying

    if augment:
        for direct in list(os.listdir('training/training/fuel_tower')):
            print(str(direct))
            data_augment(str(direct))

    images_dict = {
        'plane': list(os.listdir(training_dir + '/plane/plane')),
        'fuel_tower': list(os.listdir(training_dir + '/fuel_tower/fuel_tower')),
        # 'misc': list(os.listdir(training_dir + '/misc/misc')),
        # 'building': list(os.listdir(training_dir + '/building/building')),

    }

    label_dict = {
        'plane': 0,
        'fuel_tower': 1,
        # 'misc': 2,
        # 'building': 3,

    }
    labels = label_dict

    img_temp, name_temp = [], []
    for image_name, images in images_dict.items():
        for image in images:
            try:
                # crude fix not permanent. maybe.
                img = cv2.imread(training_dir + '/' + image_name + '/' + image_name + '/' + image)
                resized_img = cv2.resize(img, (64, 64))
                img_temp.append(resized_img)
                name_temp.append(label_dict[image_name])
            except AssertionError as msg:
                print(msg)
    img_array = np.array(img_temp, dtype=np.float32)
    name_array = np.array(name_temp, dtype=np.float32)

    # Once the dictionaries have been assembled, the program splits the content of said
    # dictionaries into a testing and training set. The image data is then normalised
    # from 0-255 pixel colour values to a range of 0-1, as to make computation later more
    # efficient, and easier to understand.

    x_train, x_test, y_train, y_test = train_test_split(img_array, name_array, random_state=0)
    print("x_train size: "+str(len(x_train)))
    print("x_test size: " + str(len(x_test)))
    train_scaled = x_train / 255
    test_scaled = x_test / 255

    return train_scaled, test_scaled, y_train, y_test, label_dict


# -------------------------------------------------------------------------------
# Recognition by a Convolutional Neural Network.
# -------------------------------------------------------------------------------

def train_cnn(images, classes, num_classes, epochs, x_test, y_test):
    data_augmentation = keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal",
                                                     input_shape=(64, 64, 3)),
        layers.experimental.preprocessing.RandomRotation(0.05),
        layers.experimental.preprocessing.RandomZoom(0.05),
    ])

    model = Sequential([
        data_augmentation,
        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                      input_shape=(64, 64, 3)),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.BatchNormalization(),
        # layers.Dropout(0.25),

        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.BatchNormalization(),

        layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPool2D(pool_size=(2, 2)),
        layers.BatchNormalization(),
        # layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')

    ])

    model.compile(optimizer=keras.optimizers.Adadelta(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(images, classes, epochs)
    print("Model Evaluation")
    performance = model.evaluate(x_test, y_test)

    return model, performance[1]


def classify(image):
    # uses the previously trained model to try and classify the extractions by
    # their content. cnns produce a probability that the output is a certain class,
    # the class with the highest probability is therefore assumed to be the object
    # within the extraction. The label of the most probable class is determined and
    # returned, as to allow the program to label the contour the extraciton is from
    # on the output image.

    try:
        global cnn_model, labels

        model = cnn_model
        image = tf.expand_dims(image, axis=0)
        cls = model.predict(image)
        most_prob = np.argmax(cls, axis=-1)
        temp = int(str(most_prob).replace("[", "").replace("]", ""))
        if cls[0][temp] < 0.6:
            label = ''
        else:
            # https://www.geeksforgeeks.org/python-get-key-from-value-in-dictionary/
            label = list(labels.keys())[list(labels.values()).index(temp)]
        return label
    except ValueError:
        print("Contour skipped; Value Error")


def main_rec():
    global training_dir, cnn_model

    training_dir = 'training/training/'
    train_sc, test_sc, train_y, test_y, labels = set_build(True)
    accuracy = 0
    while accuracy < .64:
        model, accuracy = train_cnn(train_sc, train_y, 2, 20, test_sc, test_y)

    cnn_model = model
    print("Model trained | Accuracy: "+str(accuracy))

    return labels
