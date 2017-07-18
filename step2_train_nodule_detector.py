#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 15:02:23 2017

@author: cyrano
"""

import settings
import random
import numpy
from keras.optimizers import Adam, SGD
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, merge, Convolution3D, MaxPooling3D, UpSampling3D, LeakyReLU, BatchNormalization, Flatten, Dense, Dropout, ZeroPadding3D, AveragePooling3D, Activation
from keras.models import Model, load_model, model_from_json
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from keras.utils import np_utils, plot_model
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm


#limit memory usage..
#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#set_session(tf.Session(config=config))

# zonder aug, 10:1 99 train, 97 test, 0.27 cross entropy, before commit 573
# 3 pools istead of 4 gives (bigger end layer) gives much worse validation accuray + logloss .. strange ?
# 32 x 32 x 32 lijkt het beter te doen dan 48 x 48 x 48..

K.set_image_dim_ordering("tf")
CUBE_SIZE = 32
MEAN_PIXEL_VALUE = settings.MEAN_PIXEL_VALUE_NODULE
POS_WEIGHT = 2
NEGS_PER_POS = 1
P_TH = 0.6
# POS_IMG_DIR = "luna16_train_cubes_pos"
LEARN_RATE = 0.001

USE_DROPOUT = True

def prepare_image_for_net3D(img):
    img = img.astype(numpy.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img


def data_merge(pos_with_label_list, neg_list, false_pos_list, neg_pos_rate):
    res = []
    
    res.extend(pos_with_label_list)
    
    neg_samples = []
    
#    neg_count = len(pos_with_label_list) * neg_pos_rate
    
    random.shuffle(neg_list)
    neg_samples.extend(neg_list)
    
    if false_pos_list is not None:
        random.shuffle(false_pos_list)
        neg_samples.extend(false_pos_list)
     
    print("pos_samples: ", len(pos_with_label_list), "neg_samples: ", len(neg_samples))
    
    for i, neg in enumerate(neg_samples):
        res.append((neg,0,0))        
        
    return res


def data_generator(batch_size, record_list, train_set):
    batch_idx = 0
    means = []
#    random_state = numpy.random.RandomState(1301)
    while True:
        img_list = []
        class_list = []
        size_list = []
        if train_set:
            random.shuffle(record_list)
        CROP_SIZE = CUBE_SIZE
        # CROP_SIZE = 48
        for record_idx, record_item in enumerate(record_list):
            #rint patient_dir
            class_label = record_item[1]
            size_label = record_item[2]
            if class_label == 0:
                cube_image = record_item[0]
#                cube_image = load_cube_img(record_item[0], 6, 8, 48)
                # if train_set:
                #     # helpers.save_cube_img("c:/tmp/pre.png", cube_image, 8, 8)
                #     cube_image = random_rotate_cube_img(cube_image, 0.99, -180, 180)
                #
                # if train_set:
                #     if random.randint(0, 100) > 0.1:
                #         # cube_image = numpy.flipud(cube_image)
                #         cube_image = elastic_transform48(cube_image, 64, 8, random_state)
                
                
                wiggle = 48 - CROP_SIZE - 1
                indent_x = 0
                indent_y = 0
                indent_z = 0
                if wiggle > 0:
                    indent_x = random.randint(0, wiggle)
                    indent_y = random.randint(0, wiggle)
                    indent_z = random.randint(0, wiggle)
                cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE, indent_x:indent_x + CROP_SIZE]

                if train_set:
                    if random.randint(0, 100) > 50:
                        cube_image = numpy.fliplr(cube_image)
                    if random.randint(0, 100) > 50:
                        cube_image = numpy.flipud(cube_image)
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, :, ::-1]
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, ::-1, :]

#                if CROP_SIZE != CUBE_SIZE:
#                    cube_image = rescale_patient_images2(cube_image, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
                assert cube_image.shape == (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
            else:
                cube_image = record_item[0]
#                cube_image = load_cube_img(record_item[0], 8, 8, 64)

                if train_set:
                    pass

                current_cube_size = cube_image.shape[0]
                indent_x = (current_cube_size - CROP_SIZE) / 2
                indent_y = (current_cube_size - CROP_SIZE) / 2
                indent_z = (current_cube_size - CROP_SIZE) / 2
                wiggle_indent = 0
                wiggle = current_cube_size - CROP_SIZE - 1
                if wiggle > (CROP_SIZE / 2):
                    wiggle_indent = CROP_SIZE / 4
                    wiggle = current_cube_size - CROP_SIZE - CROP_SIZE / 2 - 1
                if train_set:
                    indent_x = wiggle_indent + random.randint(0, wiggle)
                    indent_y = wiggle_indent + random.randint(0, wiggle)
                    indent_z = wiggle_indent + random.randint(0, wiggle)

                indent_x = int(indent_x)
                indent_y = int(indent_y)
                indent_z = int(indent_z)
                cube_image = cube_image[indent_z:indent_z + CROP_SIZE, indent_y:indent_y + CROP_SIZE, indent_x:indent_x + CROP_SIZE]
#                if CROP_SIZE != CUBE_SIZE:
#                    cube_image = rescale_patient_images2(cube_image, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
                assert cube_image.shape == (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)

                if train_set:
                    if random.randint(0, 100) > 50:
                        cube_image = numpy.fliplr(cube_image)
                    if random.randint(0, 100) > 50:
                        cube_image = numpy.flipud(cube_image)
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, :, ::-1]
                    if random.randint(0, 100) > 50:
                        cube_image = cube_image[:, ::-1, :]


            means.append(cube_image.mean())
            img3d = prepare_image_for_net3D(cube_image)
            if train_set:
                if len(means) % 1000000 == 0:
                    print("Mean: ", sum(means) / len(means))
            img_list.append(img3d)
            class_list.append(class_label)
            size_list.append(size_label)

            batch_idx += 1
            if batch_idx >= batch_size:
                x = numpy.vstack(img_list)
                y_class = numpy.vstack(class_list)
                y_size = numpy.vstack(size_list)
                yield x, {"out_class": y_class, "out_malignancy": y_size}
                img_list = []
                class_list = []
                size_list = []
                batch_idx = 0


def get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), load_weight_path=None, features=False, mal=False):
    inputs = Input(shape=input_shape, name="input_1")
    x = inputs
    x = AveragePooling3D(pool_size=(2, 1, 1), strides=(2, 1, 1), border_mode="same")(x)
    x = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv1', subsample=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode='valid', name='pool1')(x)

    # 2nd layer group
    x = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same', name='conv2', subsample=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool2')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.3)(x)

    # 3rd layer group
    x = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3a', subsample=(1, 1, 1))(x)
    x = Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3b', subsample=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool3')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.4)(x)

    # 4th layer group
    x = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4a', subsample=(1, 1, 1))(x)
    x = Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4b', subsample=(1, 1, 1),)(x)
    x = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool4')(x)
    if USE_DROPOUT:
        x = Dropout(p=0.5)(x)

    last64 = Convolution3D(64, 2, 2, 2, activation="relu", name="last_64")(x)
    out_class = Convolution3D(1, 1, 1, 1, activation="sigmoid", name="out_class_last")(last64)
    out_class = Flatten(name="out_class")(out_class)

    out_malignancy = Convolution3D(1, 1, 1, 1, activation=None, name="out_malignancy_last")(last64)
    out_malignancy = Flatten(name="out_malignancy")(out_malignancy)

    model = Model(input=inputs, output=[out_class, out_malignancy])
    if load_weight_path is not None:
        model.load_weights(load_weight_path, by_name=False)
    model.compile(optimizer=SGD(lr=LEARN_RATE, momentum=0.9, nesterov=True), loss={"out_class": "binary_crossentropy", "out_malignancy": mean_absolute_error}, metrics={"out_class": [binary_accuracy, binary_crossentropy], "out_malignancy": mean_absolute_error})

    if features:
        model = Model(input=inputs, output=[last64])
    model.summary(line_length=140)

    return model


def step_decay(epoch):
    res = 0.001
    if epoch > 5:
        res = 0.0001
    print("learnrate: ", res, " epoch: ", epoch)
    return res


def train(model_name, fold_count, train_full_set=False, load_weights_path=None, ndsb3_holdout=0, manual_labels=True):
    batch_size = 16
    epochs = 12
    
    train_pos = numpy.load("generated_traindata/train-pos-res.npy")        
    train_neg = numpy.load("generated_traindata/train-neg-res.npy")
    
    val_pos = numpy.load("generated_traindata/val-pos-res.npy")
    val_neg = numpy.load("generated_traindata/val-neg-res.npy")
    
    val_false_pos = numpy.load("generated_traindata/val-false-pos.npy") 
    
    print("train data: ", len(train_pos), len(train_neg), len(val_false_pos))
    print("val data: ", len(val_pos), len(val_neg))
    
#    train_neg_edge_m = int((len(train_neg_edge) * 80) / 100)
#    train_neg_lung_m = int((len(train_neg_lung) * 80) / 100)
    
#    val_false_pos_m =int((len(val_false_pos) * 80) / 100)
    
    print("trainset")
    
    train_files_res = data_merge(train_pos, train_neg, val_false_pos, NEGS_PER_POS)
    
    print("valset")
    
    holdout_files_res = data_merge(val_pos, val_neg, None, NEGS_PER_POS)
     
    print("data generator:", len(train_files_res), len(holdout_files_res))
    
    train_gen = data_generator(batch_size, train_files_res, True)
    holdout_gen = data_generator(batch_size, holdout_files_res, False)
    
    print("normalization")
    
    for i in range(0, 10):
        tmp = next(holdout_gen)
        cube_img = tmp[0][0].reshape(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1)
        cube_img = cube_img[:, :, :, 0]
        cube_img *= 255.
        cube_img += MEAN_PIXEL_VALUE
        # helpers.save_cube_img("c:/tmp/img_" + str(i) + ".png", cube_img, 4, 8)
        # print(tmp)

    learnrate_scheduler = LearningRateScheduler(step_decay)
    
    model = get_net(load_weight_path=load_weights_path)
    
    holdout_txt = "_h" + str(ndsb3_holdout) if manual_labels else ""
    if train_full_set:
        holdout_txt = "_fs" + holdout_txt
    checkpoint = ModelCheckpoint("models/model_" + model_name + "_" + holdout_txt + "_e" + "{epoch:02d}-{val_loss:.4f}.hd5", monitor='val_loss', verbose=1, save_best_only=not train_full_set, save_weights_only=False, mode='auto', period=1)
    checkpoint_fixed_name = ModelCheckpoint("models/model_" + model_name + "_" + holdout_txt + "_best.hd5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    model.fit_generator(train_gen, len(train_files_res) / 1, epochs, validation_data=holdout_gen, nb_val_samples=len(holdout_files_res) / 1, callbacks=[checkpoint, checkpoint_fixed_name, learnrate_scheduler])
    model.save("models/model_" + model_name + "_" + holdout_txt + "_end.hd5")


if __name__ == "__main__":   
#    if True:
        # model 1 on luna16 annotations. full set 1 versions for blending
        train(train_full_set=True, load_weights_path=None, model_name="final", fold_count=-1, manual_labels=False)
#        if not os.path.exists("models/"):
#            os.mkdir("models")
#        shutil.copy("workdir/model_luna16_full__fs_best.hd5", "models/model_luna16_full__fs_best.hd5")
#
#    # model 2 on luna16 annotations + ndsb pos annotations. 3 folds (1st half, 2nd half of ndsb patients) 2 versions for blending
#    if True:
#        train(train_full_set=True, load_weights_path=None, ndsb3_holdout=0, manual_labels=True, model_name="luna_posnegndsb_v1", fold_count=2)
#        train(train_full_set=True, load_weights_path=None, ndsb3_holdout=1, manual_labels=True, model_name="luna_posnegndsb_v1", fold_count=2)
#        shutil.copy("workdir/model_luna_posnegndsb_v1__fs_h0_end.hd5", "models/model_luna_posnegndsb_v1__fs_h0_end.hd5")
#        shutil.copy("workdir/model_luna_posnegndsb_v1__fs_h1_end.hd5", "models/model_luna_posnegndsb_v1__fs_h1_end.hd5")
#
#    if True:
#        train(train_full_set=True, load_weights_path=None, ndsb3_holdout=0, manual_labels=True, model_name="luna_posnegndsb_v2", fold_count=2)
#        train(train_full_set=True, load_weights_path=None, ndsb3_holdout=1, manual_labels=True, model_name="luna_posnegndsb_v2", fold_count=2)
#        shutil.copy("workdir/model_luna_posnegndsb_v2__fs_h0_end.hd5", "models/model_luna_posnegndsb_v2__fs_h0_end.hd5")
#        shutil.copy("workdir/model_luna_posnegndsb_v2__fs_h1_end.hd5", "models/model_luna_posnegndsb_v2__fs_h1_end.hd5")
