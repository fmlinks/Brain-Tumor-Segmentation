# -*- coding: utf-8 -*- 
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from keras.layers import *
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD, Adam
import numpy as np
import json
from sklearn.utils import shuffle
from scipy.ndimage.interpolation import rotate
import tensorflow as tf
from keras.applications import InceptionV3
from keras.utils import multi_gpu_model
np.random.seed(1234)

# model = InceptionV3(include_top=False, input_shape=(160, 160, 3))
# model.summary()


class DUnet0():
    def module_a(self, x, nb_filters):   # convolution
        x1 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x)
        x2 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x)
        x3 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x)

        x2_2 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', strides=1, padding='same')(x2)
        x2_2 = BatchNormalization()(x2_2)
        x3_2 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x3)

        x3_3 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x3_2)

        x = Concatenate(axis=-1)([x1, x2_2, x3_3])
        return x

    def module_b(self, x, nb_filters):   # down sample
        # input_tensor = Input(shape=(160, 160, 3))

        x = ZeroPadding2D(padding=(1, 1))(x)

        x1 = MaxPool2D(pool_size=(3, 3), strides=2)(x)        # branch1
        x2 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch2_1
        x3 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch3_1

        x2_2 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x2)       # branch2_2
        x2_2 = BatchNormalization()(x2_2)
        x3_2 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', padding='same')(x3)  # branch3_2
        x3_2 = BatchNormalization()(x3_2)
        x3_3 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x3_2)     # branch3_3

        out = Concatenate(axis=-1)([x1, x2_2, x3_3])
        return out

    def module_c(self, x, nb_filters):    # up sample
        # input_tensor = Input(shape=(160, 160, 3))

        x1 = UpSampling2D(size=(2, 2))(x)  # branch1
        x2 = Conv2DTranspose(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch2
        x3 = Conv2DTranspose(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch3

        x2_2 = Conv2DTranspose(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x2)  # branch2_2
        x3_2 = Conv2DTranspose(filters=nb_filters, kernel_size=3, activation='relu', padding='same')(x3)  # branch3_2

        x3_3 = Conv2DTranspose(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x3_2)  # branch3_3

        x2_2 = Cropping2D(cropping=((0, 1), (0, 1)))(x2_2)
        x3_3 = Cropping2D(cropping=((0, 1), (0, 1)))(x3_3)
        out = Concatenate(axis=-1)([x1, x2_2, x3_3])
        return out

    def conv_block(self, x, nb_filters, kernel=3, acti='relu'):
        x = Input(shape=x)
        blk_1 = self.module_b(x, nb_filters=32)

        blk_2 = self.module_a(blk_1, nb_filters=32)

        blk_3 = self.module_b(blk_2, nb_filters=64)

        blk_4 = self.module_a(blk_3, nb_filters=64)

        blk_5 = self.module_b(blk_4, nb_filters=128)

        blk_6 = self.module_a(blk_5, nb_filters=256)

        blk_7 = self.module_a(blk_6, nb_filters=256)

        blk_8 = self.module_c(blk_7, nb_filters=128)

        blk_8 = Concatenate(axis=-1)([blk_4, blk_8])

        blk_9 = self.module_a(blk_8, nb_filters=128)

        blk_10 = self.module_c(blk_9, nb_filters=64)

        blk_10 = Concatenate(axis=-1)([blk_2, blk_10])

        blk_11 = self.module_a(blk_10, nb_filters=64)

        blk_12 = self.module_c(blk_11, nb_filters=32)

        blk_12 = Concatenate(axis=-1)([x, blk_12])

        blk_12 = Conv2D(filters=32, kernel_size=kernel, activation='relu', padding='same')(blk_12)

        blk_12 = Conv2D(filters=16, kernel_size=kernel, activation='relu', padding='same')(blk_12)

        out = Conv2D(filters=4, kernel_size=1, activation='softmax')(blk_12)

        model = Model(x, out)
        return model

    def step_decay(slef, epochs):
        init_rate = 0.003
        fin_rate = 0.00003
        total_epochs = 24
        if epochs < 25:
            lrate = init_rate - (init_rate - fin_rate) / total_epochs * float(epochs)
        else:
            lrate = 0.00003
        return lrate

    def load_data(self, path, mode=True):
        data0 = np.load(path + 'X1.npy')
        data1 = np.load(path + 'X2.npy')
        data2 = np.load(path + 'X3.npy')
        data3 = np.load(path + 'X4.npy')

        Y_data0 = np.load(path + 'Y1.npy')
        Y_data1 = np.load(path + 'Y2.npy')
        Y_data2 = np.load(path + 'Y3.npy')
        Y_data3 = np.load(path + 'Y4.npy')

        X_train = np.concatenate((data0, data1, data2, data3), axis=0)
        Y_train = np.concatenate((Y_data0, Y_data1, Y_data2, Y_data3), axis=0)
        X_train, Y_train = shuffle(X_train, Y_train)
        print("###before data aug##", X_train.shape, Y_train.shape)

        if mode:
            X_tmp = rotate(X_train[0:7120], 90, (1, 2))  # 作旋转进行数据增强吗
            X_tmp = np.append(X_tmp, rotate(X_train[7120:], -90, (1, 2)), axis=0)
            X_train = np.append(X_train, X_tmp, axis=0)

            Y_tmp = rotate(Y_train[0:7120], 90, (1, 2))
            Y_tmp = np.append(Y_tmp, rotate(Y_train[7120:], -90, (1, 2)), axis=0)
            Y_train = np.append(Y_train, Y_tmp, axis=0)

            X_train, Y_train = shuffle(X_train, Y_train)
            print("###after data aug##", X_train.shape, Y_train.shape)
            del data0, data1, data2, data3, Y_data0, Y_data1, Y_data2, Y_data3, X_tmp, Y_tmp

        X_test = np.load(path + 'X0.npy')
        Y_test = np.load(path + 'Y0.npy')
        print(X_test.shape, Y_test.shape)
        return X_train, Y_train, X_test, Y_test

    def dice_comp(self, y_true, y_pred, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):
        y_true = y_true[:, :, :, 1:]
        y_pred = y_pred[:, :, :, 1:]
        # core_true = np.concatenate((y_true[:, :, :, 1:2], y_true[:, :, :, 3:]), axis=-1)
        # core_pred = np.concatenate((y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]), axis=-1)

        inse = tf.reduce_sum(y_pred * y_true, axis=axis)  # compute intersection
        if loss_type == 'jaccard':  # default loss type, in fact, jaccard and soresen are the same thing
            l = tf.reduce_sum(y_pred * y_pred, axis=axis)  # number of pixels in output
            r = tf.reduce_sum(y_true * y_true, axis=axis)  # number of pixels in target
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(y_pred, axis=axis)
            r = tf.reduce_sum(y_true, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)  # compute dice coefficient

        # dice1 = dice
        # dice = tf.reduce_mean(dice)  # dice coefficient is a scalar between 0 and 1
        return dice

    def dice_en(self, y_true, y_pred, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):
        y_true = y_true[:, :, :, 3:]
        y_pred = y_pred[:, :, :, 3:]
        # core_true = np.concatenate((y_true[:, :, :, 1:2], y_true[:, :, :, 3:]), axis=-1)
        # core_pred = np.concatenate((y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]), axis=-1)

        inse = tf.reduce_sum(y_pred * y_true, axis=axis)  # compute intersection
        if loss_type == 'jaccard':  # default loss type, in fact, jaccard and soresen are the same thing
            l = tf.reduce_sum(y_pred * y_pred, axis=axis)  # number of pixels in output
            r = tf.reduce_sum(y_true * y_true, axis=axis)  # number of pixels in target
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(y_pred, axis=axis)
            r = tf.reduce_sum(y_true, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)  # compute dice coefficient

        # dice1 = dice
        # dice = tf.reduce_mean(dice)  # dice coefficient is a scalar between 0 and 1
        return dice

    def dice_core(self, y_true, y_pred, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):

        y_true = Concatenate(axis=-1)([y_true[:, :, :, 1:2], y_true[:, :, :, 3:]])
        y_pred = Concatenate(axis=-1)([y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]])

        inse = tf.reduce_sum(y_pred * y_true, axis=axis)  # compute intersection
        if loss_type == 'jaccard':  # default loss type, in fact, jaccard and soresen are the same thing
            l = tf.reduce_sum(y_pred * y_pred, axis=axis)  # number of pixels in output
            r = tf.reduce_sum(y_true * y_true, axis=axis)  # number of pixels in target
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(y_pred, axis=axis)
            r = tf.reduce_sum(y_true, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)  # compute dice coefficient

        # dice1 = dice
        # dice = tf.reduce_mean(dice)  # dice coefficient is a scalar between 0 and 1
        return dice

    def train(self):
        model = self.conv_block((160, 160, 4), nb_filters=32)
        model.summary()
        # plot_model(model, to_file='./DUnet.png', show_shapes=True, show_layer_names=True)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)

        change_lr = LearningRateScheduler(self.step_decay)
        sgd = SGD(lr=0.003, momentum=0.9, decay=0, nesterov=True)
        adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        #model.load_weights('../weights/dunet_tfcv2.46-0.02.hdf5')
        #parallel_model = multi_gpu_model(model, gpus=2)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', self.dice_comp,
                                                                                  self.dice_core, self.dice_en])
        checkpointer = ModelCheckpoint(filepath='../weights/DUnet/0/dunet_tfcv5.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=70, batch_size=16,
                         shuffle=True, callbacks=[checkpointer, change_lr])
        with open('../weights/DUnet/0/dunet_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)

    def test(self):
        model = self.conv_block((240, 240, 4), nb_filters=32)
        model.summary()
        model.load_weights('./DUnet0/weights/dunet_tfcv5.70-0.02.hdf5')
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        del X_train, Y_train, Y_test
        model.summary()
        print(X_test.shape)
        pred = model.predict(X_test, verbose=1)
        np.save('./DUnet0/dunet_tfcv5_5.npy', pred)
        print(pred.shape)
        print('#### start test ###')
        pred1 = np.ndarray([pred.shape[0], pred.shape[1], pred.shape[2], 1], dtype=np.uint8)
        for slices in range(pred.shape[0]):
            if (slices % 155==0):
                print(slices)
            for i in range(pred.shape[1]):
                for j in range(pred.shape[2]):
                    pred1[slices, i, j, 0] = np.argmax(pred[slices, i, j, :])
        print(pred1.shape)
        np.save('./DUnet0/dunet_tfcv5_1.npy', pred1)

class DUnet1():
    def module_a(self, x, nb_filters):   # convolution
        x1 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x)
        x2 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x)
        x3 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x)

        x2_2 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', strides=1, padding='same')(x2)
        x2_2 = BatchNormalization()(x2_2)
        x3_2 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x3)

        x3_3 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x3_2)

        x = Concatenate(axis=-1)([x1, x2_2, x3_3])
        return x

    def module_b(self, x, nb_filters):   # down sample
        # input_tensor = Input(shape=(160, 160, 3))

        x = ZeroPadding2D(padding=(1, 1))(x)

        x1 = MaxPool2D(pool_size=(3, 3), strides=2)(x)        # branch1
        x2 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch2_1
        x3 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch3_1

        x2_2 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x2)       # branch2_2
        x2_2 = BatchNormalization()(x2_2)
        x3_2 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', padding='same')(x3)  # branch3_2
        x3_2 = BatchNormalization()(x3_2)
        x3_3 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x3_2)     # branch3_3

        out = Concatenate(axis=-1)([x1, x2_2, x3_3])
        return out

    def module_c(self, x, nb_filters):    # up sample
        # input_tensor = Input(shape=(160, 160, 3))

        x1 = UpSampling2D(size=(2, 2))(x)  # branch1
        x2 = Conv2DTranspose(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch2
        x3 = Conv2DTranspose(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch3

        x2_2 = Conv2DTranspose(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x2)  # branch2_2
        x3_2 = Conv2DTranspose(filters=nb_filters, kernel_size=3, activation='relu', padding='same')(x3)  # branch3_2

        x3_3 = Conv2DTranspose(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x3_2)  # branch3_3

        x2_2 = Cropping2D(cropping=((0, 1), (0, 1)))(x2_2)
        x3_3 = Cropping2D(cropping=((0, 1), (0, 1)))(x3_3)
        out = Concatenate(axis=-1)([x1, x2_2, x3_3])
        return out

    def conv_block(self, x, nb_filters, kernel=3, acti='relu'):
        x = Input(shape=x)
        blk_1 = self.module_b(x, nb_filters=32)

        blk_2 = self.module_a(blk_1, nb_filters=32)

        blk_3 = self.module_b(blk_2, nb_filters=64)

        blk_4 = self.module_a(blk_3, nb_filters=64)

        blk_5 = self.module_b(blk_4, nb_filters=128)

        blk_6 = self.module_a(blk_5, nb_filters=256)

        blk_7 = self.module_a(blk_6, nb_filters=256)

        blk_8 = self.module_c(blk_7, nb_filters=128)

        blk_8 = Concatenate(axis=-1)([blk_4, blk_8])

        blk_9 = self.module_a(blk_8, nb_filters=128)

        blk_10 = self.module_c(blk_9, nb_filters=64)

        blk_10 = Concatenate(axis=-1)([blk_2, blk_10])

        blk_11 = self.module_a(blk_10, nb_filters=64)

        blk_12 = self.module_c(blk_11, nb_filters=32)

        blk_12 = Concatenate(axis=-1)([x, blk_12])

        blk_12 = Conv2D(filters=32, kernel_size=kernel, activation='relu', padding='same')(blk_12)

        blk_12 = Conv2D(filters=16, kernel_size=kernel, activation='relu', padding='same')(blk_12)

        out = Conv2D(filters=4, kernel_size=1, activation='softmax')(blk_12)

        model = Model(x, out)
        return model

    def step_decay(slef, epochs):
        init_rate = 0.003
        fin_rate = 0.00003
        total_epochs = 24
        if epochs < 25:
            lrate = init_rate - (init_rate - fin_rate) / total_epochs * float(epochs)
        else:
            lrate = 0.00003
        return lrate

    def load_data(self, path, mode=True):
        data0 = np.load(path + 'X0.npy')
        data1 = np.load(path + 'X2.npy')
        data2 = np.load(path + 'X3.npy')
        data3 = np.load(path + 'X4.npy')

        Y_data0 = np.load(path + 'Y0.npy')
        Y_data1 = np.load(path + 'Y2.npy')
        Y_data2 = np.load(path + 'Y3.npy')
        Y_data3 = np.load(path + 'Y4.npy')

        X_train = np.concatenate((data0, data1, data2, data3), axis=0)
        Y_train = np.concatenate((Y_data0, Y_data1, Y_data2, Y_data3), axis=0)
        X_train, Y_train = shuffle(X_train, Y_train)
        print("###数据增强�?##", X_train.shape, Y_train.shape)

        if mode:
            X_tmp = rotate(X_train[0:7120], 90, (1, 2))  # 作旋转进行数据增强吗
            X_tmp = np.append(X_tmp, rotate(X_train[7120:], -90, (1, 2)), axis=0)
            X_train = np.append(X_train, X_tmp, axis=0)

            Y_tmp = rotate(Y_train[0:7120], 90, (1, 2))
            Y_tmp = np.append(Y_tmp, rotate(Y_train[7120:], -90, (1, 2)), axis=0)
            Y_train = np.append(Y_train, Y_tmp, axis=0)

            X_train, Y_train = shuffle(X_train, Y_train)
            print("###数据增强�?##", X_train.shape, Y_train.shape)
            del data0, data1, data2, data3, Y_data0, Y_data1, Y_data2, Y_data3, X_tmp, Y_tmp

        X_test = np.load(path + 'X1.npy')
        Y_test = np.load(path + 'Y1.npy')
        print(X_test.shape, Y_test.shape)
        return X_train, Y_train, X_test, Y_test

    def dice_comp(self, y_true, y_pred, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):
        y_true = y_true[:, :, :, 1:]
        y_pred = y_pred[:, :, :, 1:]
        # core_true = np.concatenate((y_true[:, :, :, 1:2], y_true[:, :, :, 3:]), axis=-1)
        # core_pred = np.concatenate((y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]), axis=-1)

        inse = tf.reduce_sum(y_pred * y_true, axis=axis)  # compute intersection
        if loss_type == 'jaccard':  # default loss type, in fact, jaccard and soresen are the same thing
            l = tf.reduce_sum(y_pred * y_pred, axis=axis)  # number of pixels in output
            r = tf.reduce_sum(y_true * y_true, axis=axis)  # number of pixels in target
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(y_pred, axis=axis)
            r = tf.reduce_sum(y_true, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)  # compute dice coefficient

        # dice1 = dice
        # dice = tf.reduce_mean(dice)  # dice coefficient is a scalar between 0 and 1
        return dice

    def dice_en(self, y_true, y_pred, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):
        y_true = y_true[:, :, :, 3:]
        y_pred = y_pred[:, :, :, 3:]
        # core_true = np.concatenate((y_true[:, :, :, 1:2], y_true[:, :, :, 3:]), axis=-1)
        # core_pred = np.concatenate((y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]), axis=-1)

        inse = tf.reduce_sum(y_pred * y_true, axis=axis)  # compute intersection
        if loss_type == 'jaccard':  # default loss type, in fact, jaccard and soresen are the same thing
            l = tf.reduce_sum(y_pred * y_pred, axis=axis)  # number of pixels in output
            r = tf.reduce_sum(y_true * y_true, axis=axis)  # number of pixels in target
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(y_pred, axis=axis)
            r = tf.reduce_sum(y_true, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)  # compute dice coefficient

        # dice1 = dice
        # dice = tf.reduce_mean(dice)  # dice coefficient is a scalar between 0 and 1
        return dice

    def dice_core(self, y_true, y_pred, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):

        y_true = Concatenate(axis=-1)([y_true[:, :, :, 1:2], y_true[:, :, :, 3:]])
        y_pred = Concatenate(axis=-1)([y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]])

        inse = tf.reduce_sum(y_pred * y_true, axis=axis)  # compute intersection
        if loss_type == 'jaccard':  # default loss type, in fact, jaccard and soresen are the same thing
            l = tf.reduce_sum(y_pred * y_pred, axis=axis)  # number of pixels in output
            r = tf.reduce_sum(y_true * y_true, axis=axis)  # number of pixels in target
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(y_pred, axis=axis)
            r = tf.reduce_sum(y_true, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)  # compute dice coefficient

        # dice1 = dice
        # dice = tf.reduce_mean(dice)  # dice coefficient is a scalar between 0 and 1
        return dice

    def train(self):
        model = self.conv_block((160, 160, 4), nb_filters=32)
        model.summary()
        # plot_model(model, to_file='./DUnet.png', show_shapes=True, show_layer_names=True)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)

        change_lr = LearningRateScheduler(self.step_decay)
        sgd = SGD(lr=0.003, momentum=0.9, decay=0, nesterov=True)
        adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        #model.load_weights('../weights/dunet_tfcv2.46-0.02.hdf5')
        #parallel_model = multi_gpu_model(model, gpus=2)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', self.dice_comp,
                                                                                  self.dice_core, self.dice_en])
        checkpointer = ModelCheckpoint(filepath='../weights/DUnet/1/dunet_tfcv5.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=70, batch_size=16,
                         shuffle=True, callbacks=[checkpointer, change_lr])
        with open('../weights/DUnet/1/dunet_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)

    def test(self):
        model = self.conv_block((240, 240, 4), nb_filters=32)
        model.summary()
        model.load_weights('./DUnet0/weights/dunet_tfcv5.70-0.02.hdf5')
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        del X_train, Y_train, Y_test
        model.summary()
        print(X_test.shape)
        pred = model.predict(X_test, verbose=1)
        np.save('./DUnet0/dunet_tfcv5_5.npy', pred)
        print(pred.shape)
        print('#### start test ###')
        pred1 = np.ndarray([pred.shape[0], pred.shape[1], pred.shape[2], 1], dtype=np.uint8)
        for slices in range(pred.shape[0]):
            if (slices % 155==0):
                print(slices)
            for i in range(pred.shape[1]):
                for j in range(pred.shape[2]):
                    pred1[slices, i, j, 0] = np.argmax(pred[slices, i, j, :])
        print(pred1.shape)
        np.save('./DUnet0/dunet_tfcv5_1.npy', pred1)

class DUnet2():
    def module_a(self, x, nb_filters):   # convolution
        x1 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x)
        x2 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x)
        x3 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x)

        x2_2 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', strides=1, padding='same')(x2)
        x2_2 = BatchNormalization()(x2_2)
        x3_2 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x3)

        x3_3 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x3_2)

        x = Concatenate(axis=-1)([x1, x2_2, x3_3])
        return x

    def module_b(self, x, nb_filters):   # down sample
        # input_tensor = Input(shape=(160, 160, 3))

        x = ZeroPadding2D(padding=(1, 1))(x)

        x1 = MaxPool2D(pool_size=(3, 3), strides=2)(x)        # branch1
        x2 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch2_1
        x3 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch3_1

        x2_2 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x2)       # branch2_2
        x2_2 = BatchNormalization()(x2_2)
        x3_2 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', padding='same')(x3)  # branch3_2
        x3_2 = BatchNormalization()(x3_2)
        x3_3 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x3_2)     # branch3_3

        out = Concatenate(axis=-1)([x1, x2_2, x3_3])
        return out

    def module_c(self, x, nb_filters):    # up sample
        # input_tensor = Input(shape=(160, 160, 3))

        x1 = UpSampling2D(size=(2, 2))(x)  # branch1
        x2 = Conv2DTranspose(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch2
        x3 = Conv2DTranspose(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch3

        x2_2 = Conv2DTranspose(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x2)  # branch2_2
        x3_2 = Conv2DTranspose(filters=nb_filters, kernel_size=3, activation='relu', padding='same')(x3)  # branch3_2

        x3_3 = Conv2DTranspose(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x3_2)  # branch3_3

        x2_2 = Cropping2D(cropping=((0, 1), (0, 1)))(x2_2)
        x3_3 = Cropping2D(cropping=((0, 1), (0, 1)))(x3_3)
        out = Concatenate(axis=-1)([x1, x2_2, x3_3])
        return out

    def conv_block(self, x, nb_filters, kernel=3, acti='relu'):
        x = Input(shape=x)
        blk_1 = self.module_b(x, nb_filters=32)

        blk_2 = self.module_a(blk_1, nb_filters=32)

        blk_3 = self.module_b(blk_2, nb_filters=64)

        blk_4 = self.module_a(blk_3, nb_filters=64)

        blk_5 = self.module_b(blk_4, nb_filters=128)

        blk_6 = self.module_a(blk_5, nb_filters=256)

        blk_7 = self.module_a(blk_6, nb_filters=256)

        blk_8 = self.module_c(blk_7, nb_filters=128)

        blk_8 = Concatenate(axis=-1)([blk_4, blk_8])

        blk_9 = self.module_a(blk_8, nb_filters=128)

        blk_10 = self.module_c(blk_9, nb_filters=64)

        blk_10 = Concatenate(axis=-1)([blk_2, blk_10])

        blk_11 = self.module_a(blk_10, nb_filters=64)

        blk_12 = self.module_c(blk_11, nb_filters=32)

        blk_12 = Concatenate(axis=-1)([x, blk_12])

        blk_12 = Conv2D(filters=32, kernel_size=kernel, activation='relu', padding='same')(blk_12)

        blk_12 = Conv2D(filters=16, kernel_size=kernel, activation='relu', padding='same')(blk_12)

        out = Conv2D(filters=4, kernel_size=1, activation='softmax')(blk_12)

        model = Model(x, out)
        return model

    def step_decay(slef, epochs):
        init_rate = 0.003
        fin_rate = 0.00003
        total_epochs = 24
        if epochs < 25:
            lrate = init_rate - (init_rate - fin_rate) / total_epochs * float(epochs)
        else:
            lrate = 0.00003
        return lrate

    def load_data(self, path, mode=True):
        data0 = np.load(path + 'X1.npy')
        data1 = np.load(path + 'X0.npy')
        data2 = np.load(path + 'X3.npy')
        data3 = np.load(path + 'X4.npy')

        Y_data0 = np.load(path + 'Y1.npy')
        Y_data1 = np.load(path + 'Y0.npy')
        Y_data2 = np.load(path + 'Y3.npy')
        Y_data3 = np.load(path + 'Y4.npy')

        X_train = np.concatenate((data0, data1, data2, data3), axis=0)
        Y_train = np.concatenate((Y_data0, Y_data1, Y_data2, Y_data3), axis=0)
        X_train, Y_train = shuffle(X_train, Y_train)
        print("###数据增强�?##", X_train.shape, Y_train.shape)

        if mode:
            X_tmp = rotate(X_train[0:7120], 90, (1, 2))  # 作旋转进行数据增强吗
            X_tmp = np.append(X_tmp, rotate(X_train[7120:], -90, (1, 2)), axis=0)
            X_train = np.append(X_train, X_tmp, axis=0)

            Y_tmp = rotate(Y_train[0:7120], 90, (1, 2))
            Y_tmp = np.append(Y_tmp, rotate(Y_train[7120:], -90, (1, 2)), axis=0)
            Y_train = np.append(Y_train, Y_tmp, axis=0)

            X_train, Y_train = shuffle(X_train, Y_train)
            print("###数据增强�?##", X_train.shape, Y_train.shape)
            del data0, data1, data2, data3, Y_data0, Y_data1, Y_data2, Y_data3, X_tmp, Y_tmp

        X_test = np.load(path + 'X2.npy')
        Y_test = np.load(path + 'Y2.npy')
        print(X_test.shape, Y_test.shape)
        return X_train, Y_train, X_test, Y_test

    def dice_comp(self, y_true, y_pred, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):
        y_true = y_true[:, :, :, 1:]
        y_pred = y_pred[:, :, :, 1:]
        # core_true = np.concatenate((y_true[:, :, :, 1:2], y_true[:, :, :, 3:]), axis=-1)
        # core_pred = np.concatenate((y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]), axis=-1)

        inse = tf.reduce_sum(y_pred * y_true, axis=axis)  # compute intersection
        if loss_type == 'jaccard':  # default loss type, in fact, jaccard and soresen are the same thing
            l = tf.reduce_sum(y_pred * y_pred, axis=axis)  # number of pixels in output
            r = tf.reduce_sum(y_true * y_true, axis=axis)  # number of pixels in target
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(y_pred, axis=axis)
            r = tf.reduce_sum(y_true, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)  # compute dice coefficient

        # dice1 = dice
        # dice = tf.reduce_mean(dice)  # dice coefficient is a scalar between 0 and 1
        return dice

    def dice_en(self, y_true, y_pred, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):
        y_true = y_true[:, :, :, 3:]
        y_pred = y_pred[:, :, :, 3:]
        # core_true = np.concatenate((y_true[:, :, :, 1:2], y_true[:, :, :, 3:]), axis=-1)
        # core_pred = np.concatenate((y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]), axis=-1)

        inse = tf.reduce_sum(y_pred * y_true, axis=axis)  # compute intersection
        if loss_type == 'jaccard':  # default loss type, in fact, jaccard and soresen are the same thing
            l = tf.reduce_sum(y_pred * y_pred, axis=axis)  # number of pixels in output
            r = tf.reduce_sum(y_true * y_true, axis=axis)  # number of pixels in target
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(y_pred, axis=axis)
            r = tf.reduce_sum(y_true, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)  # compute dice coefficient

        # dice1 = dice
        # dice = tf.reduce_mean(dice)  # dice coefficient is a scalar between 0 and 1
        return dice

    def dice_core(self, y_true, y_pred, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):

        y_true = Concatenate(axis=-1)([y_true[:, :, :, 1:2], y_true[:, :, :, 3:]])
        y_pred = Concatenate(axis=-1)([y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]])

        inse = tf.reduce_sum(y_pred * y_true, axis=axis)  # compute intersection
        if loss_type == 'jaccard':  # default loss type, in fact, jaccard and soresen are the same thing
            l = tf.reduce_sum(y_pred * y_pred, axis=axis)  # number of pixels in output
            r = tf.reduce_sum(y_true * y_true, axis=axis)  # number of pixels in target
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(y_pred, axis=axis)
            r = tf.reduce_sum(y_true, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)  # compute dice coefficient

        # dice1 = dice
        # dice = tf.reduce_mean(dice)  # dice coefficient is a scalar between 0 and 1
        return dice

    def train(self):
        model = self.conv_block((160, 160, 4), nb_filters=32)
        model.summary()
        # plot_model(model, to_file='./DUnet.png', show_shapes=True, show_layer_names=True)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)

        change_lr = LearningRateScheduler(self.step_decay)
        sgd = SGD(lr=0.003, momentum=0.9, decay=0, nesterov=True)
        adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        #model.load_weights('../weights/dunet_tfcv2.46-0.02.hdf5')
        #parallel_model = multi_gpu_model(model, gpus=2)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', self.dice_comp,
                                                                                  self.dice_core, self.dice_en])
        checkpointer = ModelCheckpoint(filepath='../weights/DUnet/2/dunet_tfcv5.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=70, batch_size=16,
                         shuffle=True, callbacks=[checkpointer, change_lr])
        with open('../weights/DUnet/2/dunet_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)

    def test(self):
        model = self.conv_block((240, 240, 4), nb_filters=32)
        model.summary()
        model.load_weights('./DUnet0/weights/dunet_tfcv5.70-0.02.hdf5')
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        del X_train, Y_train, Y_test
        model.summary()
        print(X_test.shape)
        pred = model.predict(X_test, verbose=1)
        np.save('./DUnet0/dunet_tfcv5_5.npy', pred)
        print(pred.shape)
        print('#### start test ###')
        pred1 = np.ndarray([pred.shape[0], pred.shape[1], pred.shape[2], 1], dtype=np.uint8)
        for slices in range(pred.shape[0]):
            if (slices % 155==0):
                print(slices)
            for i in range(pred.shape[1]):
                for j in range(pred.shape[2]):
                    pred1[slices, i, j, 0] = np.argmax(pred[slices, i, j, :])
        print(pred1.shape)
        np.save('./DUnet0/dunet_tfcv5_1.npy', pred1)

class DUnet3():
    def module_a(self, x, nb_filters):   # convolution
        x1 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x)
        x2 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x)
        x3 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x)

        x2_2 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', strides=1, padding='same')(x2)
        x2_2 = BatchNormalization()(x2_2)
        x3_2 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x3)

        x3_3 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x3_2)

        x = Concatenate(axis=-1)([x1, x2_2, x3_3])
        return x

    def module_b(self, x, nb_filters):   # down sample
        # input_tensor = Input(shape=(160, 160, 3))

        x = ZeroPadding2D(padding=(1, 1))(x)

        x1 = MaxPool2D(pool_size=(3, 3), strides=2)(x)        # branch1
        x2 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch2_1
        x3 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch3_1

        x2_2 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x2)       # branch2_2
        x2_2 = BatchNormalization()(x2_2)
        x3_2 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', padding='same')(x3)  # branch3_2
        x3_2 = BatchNormalization()(x3_2)
        x3_3 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x3_2)     # branch3_3

        out = Concatenate(axis=-1)([x1, x2_2, x3_3])
        return out

    def module_c(self, x, nb_filters):    # up sample
        # input_tensor = Input(shape=(160, 160, 3))

        x1 = UpSampling2D(size=(2, 2))(x)  # branch1
        x2 = Conv2DTranspose(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch2
        x3 = Conv2DTranspose(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch3

        x2_2 = Conv2DTranspose(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x2)  # branch2_2
        x3_2 = Conv2DTranspose(filters=nb_filters, kernel_size=3, activation='relu', padding='same')(x3)  # branch3_2

        x3_3 = Conv2DTranspose(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x3_2)  # branch3_3

        x2_2 = Cropping2D(cropping=((0, 1), (0, 1)))(x2_2)
        x3_3 = Cropping2D(cropping=((0, 1), (0, 1)))(x3_3)
        out = Concatenate(axis=-1)([x1, x2_2, x3_3])
        return out

    def conv_block(self, x, nb_filters, kernel=3, acti='relu'):
        x = Input(shape=x)
        blk_1 = self.module_b(x, nb_filters=32)

        blk_2 = self.module_a(blk_1, nb_filters=32)

        blk_3 = self.module_b(blk_2, nb_filters=64)

        blk_4 = self.module_a(blk_3, nb_filters=64)

        blk_5 = self.module_b(blk_4, nb_filters=128)

        blk_6 = self.module_a(blk_5, nb_filters=256)

        blk_7 = self.module_a(blk_6, nb_filters=256)

        blk_8 = self.module_c(blk_7, nb_filters=128)

        blk_8 = Concatenate(axis=-1)([blk_4, blk_8])

        blk_9 = self.module_a(blk_8, nb_filters=128)

        blk_10 = self.module_c(blk_9, nb_filters=64)

        blk_10 = Concatenate(axis=-1)([blk_2, blk_10])

        blk_11 = self.module_a(blk_10, nb_filters=64)

        blk_12 = self.module_c(blk_11, nb_filters=32)

        blk_12 = Concatenate(axis=-1)([x, blk_12])

        blk_12 = Conv2D(filters=32, kernel_size=kernel, activation='relu', padding='same')(blk_12)

        blk_12 = Conv2D(filters=16, kernel_size=kernel, activation='relu', padding='same')(blk_12)

        out = Conv2D(filters=4, kernel_size=1, activation='softmax')(blk_12)

        model = Model(x, out)
        return model

    def step_decay(slef, epochs):
        init_rate = 0.003
        fin_rate = 0.00003
        total_epochs = 24
        if epochs < 25:
            lrate = init_rate - (init_rate - fin_rate) / total_epochs * float(epochs)
        else:
            lrate = 0.00003
        return lrate

    def load_data(self, path, mode=True):
        data0 = np.load(path + 'X1.npy')
        data1 = np.load(path + 'X2.npy')
        data2 = np.load(path + 'X0.npy')
        data3 = np.load(path + 'X4.npy')

        Y_data0 = np.load(path + 'Y1.npy')
        Y_data1 = np.load(path + 'Y2.npy')
        Y_data2 = np.load(path + 'Y0.npy')
        Y_data3 = np.load(path + 'Y4.npy')

        X_train = np.concatenate((data0, data1, data2, data3), axis=0)
        Y_train = np.concatenate((Y_data0, Y_data1, Y_data2, Y_data3), axis=0)
        X_train, Y_train = shuffle(X_train, Y_train)
        print("###数据增强�?##", X_train.shape, Y_train.shape)

        if mode:
            X_tmp = rotate(X_train[0:7120], 90, (1, 2))  # 作旋转进行数据增强吗
            X_tmp = np.append(X_tmp, rotate(X_train[7120:], -90, (1, 2)), axis=0)
            X_train = np.append(X_train, X_tmp, axis=0)

            Y_tmp = rotate(Y_train[0:7120], 90, (1, 2))
            Y_tmp = np.append(Y_tmp, rotate(Y_train[7120:], -90, (1, 2)), axis=0)
            Y_train = np.append(Y_train, Y_tmp, axis=0)

            X_train, Y_train = shuffle(X_train, Y_train)
            print("###数据增强�?##", X_train.shape, Y_train.shape)
            del data0, data1, data2, data3, Y_data0, Y_data1, Y_data2, Y_data3, X_tmp, Y_tmp

        X_test = np.load(path + 'X3.npy')
        Y_test = np.load(path + 'Y3.npy')
        print(X_test.shape, Y_test.shape)
        return X_train, Y_train, X_test, Y_test

    def dice_comp(self, y_true, y_pred, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):
        y_true = y_true[:, :, :, 1:]
        y_pred = y_pred[:, :, :, 1:]
        # core_true = np.concatenate((y_true[:, :, :, 1:2], y_true[:, :, :, 3:]), axis=-1)
        # core_pred = np.concatenate((y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]), axis=-1)

        inse = tf.reduce_sum(y_pred * y_true, axis=axis)  # compute intersection
        if loss_type == 'jaccard':  # default loss type, in fact, jaccard and soresen are the same thing
            l = tf.reduce_sum(y_pred * y_pred, axis=axis)  # number of pixels in output
            r = tf.reduce_sum(y_true * y_true, axis=axis)  # number of pixels in target
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(y_pred, axis=axis)
            r = tf.reduce_sum(y_true, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)  # compute dice coefficient

        # dice1 = dice
        # dice = tf.reduce_mean(dice)  # dice coefficient is a scalar between 0 and 1
        return dice

    def dice_en(self, y_true, y_pred, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):
        y_true = y_true[:, :, :, 3:]
        y_pred = y_pred[:, :, :, 3:]
        # core_true = np.concatenate((y_true[:, :, :, 1:2], y_true[:, :, :, 3:]), axis=-1)
        # core_pred = np.concatenate((y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]), axis=-1)

        inse = tf.reduce_sum(y_pred * y_true, axis=axis)  # compute intersection
        if loss_type == 'jaccard':  # default loss type, in fact, jaccard and soresen are the same thing
            l = tf.reduce_sum(y_pred * y_pred, axis=axis)  # number of pixels in output
            r = tf.reduce_sum(y_true * y_true, axis=axis)  # number of pixels in target
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(y_pred, axis=axis)
            r = tf.reduce_sum(y_true, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)  # compute dice coefficient

        # dice1 = dice
        # dice = tf.reduce_mean(dice)  # dice coefficient is a scalar between 0 and 1
        return dice

    def dice_core(self, y_true, y_pred, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):

        y_true = Concatenate(axis=-1)([y_true[:, :, :, 1:2], y_true[:, :, :, 3:]])
        y_pred = Concatenate(axis=-1)([y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]])

        inse = tf.reduce_sum(y_pred * y_true, axis=axis)  # compute intersection
        if loss_type == 'jaccard':  # default loss type, in fact, jaccard and soresen are the same thing
            l = tf.reduce_sum(y_pred * y_pred, axis=axis)  # number of pixels in output
            r = tf.reduce_sum(y_true * y_true, axis=axis)  # number of pixels in target
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(y_pred, axis=axis)
            r = tf.reduce_sum(y_true, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)  # compute dice coefficient

        # dice1 = dice
        # dice = tf.reduce_mean(dice)  # dice coefficient is a scalar between 0 and 1
        return dice

    def train(self):
        model = self.conv_block((160, 160, 4), nb_filters=32)
        model.summary()
        # plot_model(model, to_file='./DUnet.png', show_shapes=True, show_layer_names=True)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)

        change_lr = LearningRateScheduler(self.step_decay)
        sgd = SGD(lr=0.003, momentum=0.9, decay=0, nesterov=True)
        adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        #model.load_weights('../weights/dunet_tfcv2.46-0.02.hdf5')
        #parallel_model = multi_gpu_model(model, gpus=2)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', self.dice_comp,
                                                                                  self.dice_core, self.dice_en])
        checkpointer = ModelCheckpoint(filepath='../weights/DUnet/3/dunet_tfcv5.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=70, batch_size=16,
                         shuffle=True, callbacks=[checkpointer, change_lr])
        with open('../weights/DUnet/3/dunet_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)

    def test(self):
        model = self.conv_block((240, 240, 4), nb_filters=32)
        model.summary()
        model.load_weights('./DUnet0/weights/dunet_tfcv5.70-0.02.hdf5')
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        del X_train, Y_train, Y_test
        model.summary()
        print(X_test.shape)
        pred = model.predict(X_test, verbose=1)
        np.save('./DUnet0/dunet_tfcv5_5.npy', pred)
        print(pred.shape)
        print('#### start test ###')
        pred1 = np.ndarray([pred.shape[0], pred.shape[1], pred.shape[2], 1], dtype=np.uint8)
        for slices in range(pred.shape[0]):
            if (slices % 155==0):
                print(slices)
            for i in range(pred.shape[1]):
                for j in range(pred.shape[2]):
                    pred1[slices, i, j, 0] = np.argmax(pred[slices, i, j, :])
        print(pred1.shape)
        np.save('./DUnet0/dunet_tfcv5_1.npy', pred1)

class DUnet4():
    def module_a(self, x, nb_filters):   # convolution
        x1 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x)
        x2 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x)
        x3 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x)

        x2_2 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', strides=1, padding='same')(x2)
        x2_2 = BatchNormalization()(x2_2)
        x3_2 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x3)

        x3_3 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1, padding='same')(x3_2)

        x = Concatenate(axis=-1)([x1, x2_2, x3_3])
        return x

    def module_b(self, x, nb_filters):   # down sample
        # input_tensor = Input(shape=(160, 160, 3))

        x = ZeroPadding2D(padding=(1, 1))(x)

        x1 = MaxPool2D(pool_size=(3, 3), strides=2)(x)        # branch1
        x2 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch2_1
        x3 = Conv2D(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch3_1

        x2_2 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x2)       # branch2_2
        x2_2 = BatchNormalization()(x2_2)
        x3_2 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', padding='same')(x3)  # branch3_2
        x3_2 = BatchNormalization()(x3_2)
        x3_3 = Conv2D(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x3_2)     # branch3_3

        out = Concatenate(axis=-1)([x1, x2_2, x3_3])
        return out

    def module_c(self, x, nb_filters):    # up sample
        # input_tensor = Input(shape=(160, 160, 3))

        x1 = UpSampling2D(size=(2, 2))(x)  # branch1
        x2 = Conv2DTranspose(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch2
        x3 = Conv2DTranspose(filters=nb_filters, kernel_size=1, activation='relu', strides=1)(x)  # branch3

        x2_2 = Conv2DTranspose(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x2)  # branch2_2
        x3_2 = Conv2DTranspose(filters=nb_filters, kernel_size=3, activation='relu', padding='same')(x3)  # branch3_2

        x3_3 = Conv2DTranspose(filters=nb_filters, kernel_size=3, activation='relu', strides=2)(x3_2)  # branch3_3

        x2_2 = Cropping2D(cropping=((0, 1), (0, 1)))(x2_2)
        x3_3 = Cropping2D(cropping=((0, 1), (0, 1)))(x3_3)
        out = Concatenate(axis=-1)([x1, x2_2, x3_3])
        return out

    def conv_block(self, x, nb_filters, kernel=3, acti='relu'):
        x = Input(shape=x)
        blk_1 = self.module_b(x, nb_filters=32)

        blk_2 = self.module_a(blk_1, nb_filters=32)

        blk_3 = self.module_b(blk_2, nb_filters=64)

        blk_4 = self.module_a(blk_3, nb_filters=64)

        blk_5 = self.module_b(blk_4, nb_filters=128)

        blk_6 = self.module_a(blk_5, nb_filters=256)

        blk_7 = self.module_a(blk_6, nb_filters=256)

        blk_8 = self.module_c(blk_7, nb_filters=128)

        blk_8 = Concatenate(axis=-1)([blk_4, blk_8])

        blk_9 = self.module_a(blk_8, nb_filters=128)

        blk_10 = self.module_c(blk_9, nb_filters=64)

        blk_10 = Concatenate(axis=-1)([blk_2, blk_10])

        blk_11 = self.module_a(blk_10, nb_filters=64)

        blk_12 = self.module_c(blk_11, nb_filters=32)

        blk_12 = Concatenate(axis=-1)([x, blk_12])

        blk_12 = Conv2D(filters=32, kernel_size=kernel, activation='relu', padding='same')(blk_12)

        blk_12 = Conv2D(filters=16, kernel_size=kernel, activation='relu', padding='same')(blk_12)

        out = Conv2D(filters=4, kernel_size=1, activation='softmax')(blk_12)

        model = Model(x, out)
        return model

    def step_decay(slef, epochs):
        init_rate = 0.003
        fin_rate = 0.00003
        total_epochs = 24
        if epochs < 25:
            lrate = init_rate - (init_rate - fin_rate) / total_epochs * float(epochs)
        else:
            lrate = 0.00003
        return lrate

    def load_data(self, path, mode=True):
        data0 = np.load(path + 'X1.npy')
        data1 = np.load(path + 'X2.npy')
        data2 = np.load(path + 'X3.npy')
        data3 = np.load(path + 'X0.npy')

        Y_data0 = np.load(path + 'Y1.npy')
        Y_data1 = np.load(path + 'Y2.npy')
        Y_data2 = np.load(path + 'Y3.npy')
        Y_data3 = np.load(path + 'Y0.npy')

        X_train = np.concatenate((data0, data1, data2, data3), axis=0)
        Y_train = np.concatenate((Y_data0, Y_data1, Y_data2, Y_data3), axis=0)
        X_train, Y_train = shuffle(X_train, Y_train)
        print("###数据增强�?##", X_train.shape, Y_train.shape)

        if mode:
            X_tmp = rotate(X_train[0:7120], 90, (1, 2))  # 作旋转进行数据增强吗
            X_tmp = np.append(X_tmp, rotate(X_train[7120:], -90, (1, 2)), axis=0)
            X_train = np.append(X_train, X_tmp, axis=0)

            Y_tmp = rotate(Y_train[0:7120], 90, (1, 2))
            Y_tmp = np.append(Y_tmp, rotate(Y_train[7120:], -90, (1, 2)), axis=0)
            Y_train = np.append(Y_train, Y_tmp, axis=0)

            X_train, Y_train = shuffle(X_train, Y_train)
            print("###数据增强�?##", X_train.shape, Y_train.shape)
            del data0, data1, data2, data3, Y_data0, Y_data1, Y_data2, Y_data3, X_tmp, Y_tmp

        X_test = np.load(path + 'X4.npy')
        Y_test = np.load(path + 'Y4.npy')
        print(X_test.shape, Y_test.shape)
        return X_train, Y_train, X_test, Y_test

    def dice_comp(self, y_true, y_pred, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):
        y_true = y_true[:, :, :, 1:]
        y_pred = y_pred[:, :, :, 1:]
        # core_true = np.concatenate((y_true[:, :, :, 1:2], y_true[:, :, :, 3:]), axis=-1)
        # core_pred = np.concatenate((y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]), axis=-1)

        inse = tf.reduce_sum(y_pred * y_true, axis=axis)  # compute intersection
        if loss_type == 'jaccard':  # default loss type, in fact, jaccard and soresen are the same thing
            l = tf.reduce_sum(y_pred * y_pred, axis=axis)  # number of pixels in output
            r = tf.reduce_sum(y_true * y_true, axis=axis)  # number of pixels in target
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(y_pred, axis=axis)
            r = tf.reduce_sum(y_true, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)  # compute dice coefficient

        # dice1 = dice
        # dice = tf.reduce_mean(dice)  # dice coefficient is a scalar between 0 and 1
        return dice

    def dice_en(self, y_true, y_pred, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):
        y_true = y_true[:, :, :, 3:]
        y_pred = y_pred[:, :, :, 3:]
        # core_true = np.concatenate((y_true[:, :, :, 1:2], y_true[:, :, :, 3:]), axis=-1)
        # core_pred = np.concatenate((y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]), axis=-1)

        inse = tf.reduce_sum(y_pred * y_true, axis=axis)  # compute intersection
        if loss_type == 'jaccard':  # default loss type, in fact, jaccard and soresen are the same thing
            l = tf.reduce_sum(y_pred * y_pred, axis=axis)  # number of pixels in output
            r = tf.reduce_sum(y_true * y_true, axis=axis)  # number of pixels in target
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(y_pred, axis=axis)
            r = tf.reduce_sum(y_true, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)  # compute dice coefficient

        # dice1 = dice
        # dice = tf.reduce_mean(dice)  # dice coefficient is a scalar between 0 and 1
        return dice

    def dice_core(self, y_true, y_pred, loss_type='jaccard', axis=[1, 2, 3], smooth=1e-5):

        y_true = Concatenate(axis=-1)([y_true[:, :, :, 1:2], y_true[:, :, :, 3:]])
        y_pred = Concatenate(axis=-1)([y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]])

        inse = tf.reduce_sum(y_pred * y_true, axis=axis)  # compute intersection
        if loss_type == 'jaccard':  # default loss type, in fact, jaccard and soresen are the same thing
            l = tf.reduce_sum(y_pred * y_pred, axis=axis)  # number of pixels in output
            r = tf.reduce_sum(y_true * y_true, axis=axis)  # number of pixels in target
        elif loss_type == 'sorensen':
            l = tf.reduce_sum(y_pred, axis=axis)
            r = tf.reduce_sum(y_true, axis=axis)
        else:
            raise Exception("Unknow loss_type")
        dice = (2. * inse + smooth) / (l + r + smooth)  # compute dice coefficient

        # dice1 = dice
        # dice = tf.reduce_mean(dice)  # dice coefficient is a scalar between 0 and 1
        return dice

    def train(self):
        model = self.conv_block((160, 160, 4), nb_filters=32)
        model.summary()
        # plot_model(model, to_file='./DUnet.png', show_shapes=True, show_layer_names=True)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)

        change_lr = LearningRateScheduler(self.step_decay)
        sgd = SGD(lr=0.003, momentum=0.9, decay=0, nesterov=True)
        adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        #model.load_weights('../weights/dunet_tfcv2.46-0.02.hdf5')
        #parallel_model = multi_gpu_model(model, gpus=2)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', self.dice_comp,
                                                                                  self.dice_core, self.dice_en])
        checkpointer = ModelCheckpoint(filepath='../weights/DUnet/4/dunet_tfcv5.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=70, batch_size=16,
                         shuffle=True, callbacks=[checkpointer, change_lr])
        with open('../weights/DUnet/4/dunet_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)

    def test(self):
        model = self.conv_block((240, 240, 4), nb_filters=32)
        model.summary()
        model.load_weights('./DUnet0/weights/dunet_tfcv5.70-0.02.hdf5')
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        del X_train, Y_train, Y_test
        model.summary()
        print(X_test.shape)
        pred = model.predict(X_test, verbose=1)
        np.save('./DUnet0/dunet_tfcv5_5.npy', pred)
        print(pred.shape)
        print('#### start test ###')
        pred1 = np.ndarray([pred.shape[0], pred.shape[1], pred.shape[2], 1], dtype=np.uint8)
        for slices in range(pred.shape[0]):
            if (slices % 155==0):
                print(slices)
            for i in range(pred.shape[1]):
                for j in range(pred.shape[2]):
                    pred1[slices, i, j, 0] = np.argmax(pred[slices, i, j, :])
        print(pred1.shape)
        np.save('./DUnet0/dunet_tfcv5_1.npy', pred1)

if __name__ == '__main__':
    # DUnet0 = DUnet0()
    # DUnet0.train()

    DUnet1 = DUnet1()
    DUnet1.train()

    DUnet2 = DUnet2()
    DUnet2.train()

    DUnet3 = DUnet3()
    DUnet3.train()

    DUnet4 = DUnet4()
    DUnet4.train()
