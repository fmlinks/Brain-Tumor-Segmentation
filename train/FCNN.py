# Brain Tumor Segmentation from Multi Modal MR images using Fully Convolutional Neural Network
# 其实就是UNET, 作者改名为FCNN

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from keras.layers import Conv2D, MaxPool2D, UpSampling2D, Input, Deconv2D, Conv2DTranspose, \
    Concatenate, BatchNormalization, Activation
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD, Adam
import tensorflow as tf
import numpy as np
from scipy.ndimage.interpolation import rotate
import json
from sklearn.utils import shuffle
np.random.seed(1234)


class FCNN0():
    def conv_block(self, x, nb_filters, kernel, depth, factor, acti):
        if depth > 0:
            m = Conv2D(nb_filters, kernel, padding='same')(x)
            m = BatchNormalization()(m)
            x = Activation(activation=acti)(m)

            x = Conv2D(nb_filters, kernel, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation(activation=acti)(x)
            x = MaxPool2D()(x)

            x = self.conv_block(x, int(nb_filters*factor), kernel, depth-1, factor, acti)
            x = UpSampling2D()(x)
            x = Conv2D(nb_filters, kernel, activation=acti, padding='same')(x)
            # x = Conv2DTranspose(nb_filters, kernel, strides=2, activation=acti)(x)
            x = Concatenate(axis=-1)([x, m])
        x = Conv2D(nb_filters, kernel, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation=acti)(x)
        x = Conv2D(nb_filters, kernel, padding='same')(x)
        x = BatchNormalization()(x)
        return Activation(activation=acti)(x)

    def net(self, img_shape, n_out=4, dim=32, depth=4, factor=2, acti='relu', flatten=False):
        i = Input(shape=img_shape)
        o = self.conv_block(i, nb_filters=dim, kernel=3, depth=depth, factor=factor, acti=acti)
        o = Conv2D(n_out, (1, 1))(o)
        # if flatten:
        #     o = Reshape(n_out, img_shape[0] * img_shape[1])(o)
        #     o = Permute((2, 1))(o)
        o = Activation('softmax')(o)
        return Model(inputs=i, outputs=o)

    def step_decay(slef, epochs):
        init_rate = 0.001
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
        print("###数据增强前###", X_train.shape, Y_train.shape)

        if mode:
            X_tmp = rotate(X_train[0:7500], 90, (1, 2))  # 作旋转进行数据增强吗
            X_tmp = np.append(X_tmp, rotate(X_train[7500:], -90, (1, 2)), axis=0)
            X_train = np.append(X_train, X_tmp, axis=0)

            Y_tmp = rotate(Y_train[0:7500], 90, (1, 2))
            Y_tmp = np.append(Y_tmp, rotate(Y_train[7500:], -90, (1, 2)), axis=0)
            Y_train = np.append(Y_train, Y_tmp, axis=0)

            X_train, Y_train = shuffle(X_train, Y_train)
            print("###数据增强后###", X_train.shape, Y_train.shape)
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
        # y_true = y_true[:, :, :, 4:]
        # y_pred = y_pred[:, :, :, 4:]

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

        # y_true = Concatenate(axis=-1)([y_true[:, :, :, 1:2], y_true[:, :, :, 3:]])
        # y_pred = Concatenate(axis=-1)([y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]])
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
        # fcnn = FCNN()
        model = self.net((240, 240, 4), dim=32, factor=2)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        model.summary()

        change_lr = LearningRateScheduler(self.step_decay)
        sgd = SGD(lr=0.003, momentum=0.9, decay=0, nesterov=True)
        adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', self.dice_comp,
                                                                                  self.dice_core, self.dice_en])
        checkpointer = ModelCheckpoint(filepath='../weights/FCNN/0/deconv_150_5.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=70, batch_size=10,
                         shuffle=True, callbacks=[checkpointer, change_lr])
        with open('../weights/FCNN/0/fcnn_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)

    def test(self):
        model = self.net((240, 240, 4), dim=32, factor=2)
        model.load_weights('./FCNN0/weights/deconv_150_5.69-0.02.hdf5')
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        del X_train, Y_train, Y_test
        model.summary()
        print(X_test.shape)
        pred = model.predict(X_test, verbose=1)
        np.save('./FCNN0/fcnn_tfcv5_5.npy', pred)
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
        np.save('./fcnn_tfcv5_1.npy', pred1)


class FCNN1():
    def conv_block(self, x, nb_filters, kernel, depth, factor, acti):
        if depth > 0:
            m = Conv2D(nb_filters, kernel, padding='same')(x)
            m = BatchNormalization()(m)
            x = Activation(activation=acti)(m)

            x = Conv2D(nb_filters, kernel, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation(activation=acti)(x)
            x = MaxPool2D()(x)

            x = self.conv_block(x, int(nb_filters * factor), kernel, depth - 1, factor, acti)
            x = UpSampling2D()(x)
            x = Conv2D(nb_filters, kernel, activation=acti, padding='same')(x)
            # x = Conv2DTranspose(nb_filters, kernel, strides=2, activation=acti)(x)
            x = Concatenate(axis=-1)([x, m])
        x = Conv2D(nb_filters, kernel, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation=acti)(x)
        x = Conv2D(nb_filters, kernel, padding='same')(x)
        x = BatchNormalization()(x)
        return Activation(activation=acti)(x)

    def net(self, img_shape, n_out=4, dim=32, depth=4, factor=2, acti='relu', flatten=False):
        i = Input(shape=img_shape)
        o = self.conv_block(i, nb_filters=dim, kernel=3, depth=depth, factor=factor, acti=acti)
        o = Conv2D(n_out, (1, 1))(o)
        # if flatten:
        #     o = Reshape(n_out, img_shape[0] * img_shape[1])(o)
        #     o = Permute((2, 1))(o)
        o = Activation('softmax')(o)
        return Model(inputs=i, outputs=o)

    def step_decay(slef, epochs):
        init_rate = 0.001
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
        print("###数据增强前###", X_train.shape, Y_train.shape)

        if mode:
            X_tmp = rotate(X_train[0:7500], 90, (1, 2))  # 作旋转进行数据增强吗
            X_tmp = np.append(X_tmp, rotate(X_train[7500:], -90, (1, 2)), axis=0)
            X_train = np.append(X_train, X_tmp, axis=0)

            Y_tmp = rotate(Y_train[0:7500], 90, (1, 2))
            Y_tmp = np.append(Y_tmp, rotate(Y_train[7500:], -90, (1, 2)), axis=0)
            Y_train = np.append(Y_train, Y_tmp, axis=0)

            X_train, Y_train = shuffle(X_train, Y_train)
            print("###数据增强后###", X_train.shape, Y_train.shape)
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
        # y_true = y_true[:, :, :, 4:]
        # y_pred = y_pred[:, :, :, 4:]

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

        # y_true = Concatenate(axis=-1)([y_true[:, :, :, 1:2], y_true[:, :, :, 3:]])
        # y_pred = Concatenate(axis=-1)([y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]])
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
        # fcnn = FCNN()
        model = self.net((240, 240, 4), dim=32, factor=2)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        model.summary()

        change_lr = LearningRateScheduler(self.step_decay)
        sgd = SGD(lr=0.003, momentum=0.9, decay=0, nesterov=True)
        adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', self.dice_comp,
                                                                                  self.dice_core, self.dice_en])
        checkpointer = ModelCheckpoint(filepath='../weights/FCNN/1/deconv_150_5.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=70, batch_size=10,
                         shuffle=True, callbacks=[checkpointer, change_lr])
        with open('../weights/FCNN/1/fcnn_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)

    def test(self):
        model = self.net((240, 240, 4), dim=32, factor=2)
        model.load_weights('./FCNN1/weights/deconv_150_5.69-0.02.hdf5')
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        del X_train, Y_train, Y_test
        model.summary()
        print(X_test.shape)
        pred = model.predict(X_test, verbose=1)
        np.save('./FCNN1/fcnn_tfcv5_5.npy', pred)
        print(pred.shape)
        print('#### start test ###')
        pred1 = np.ndarray([pred.shape[0], pred.shape[1], pred.shape[2], 1], dtype=np.uint8)
        for slices in range(pred.shape[0]):
            if (slices % 155 == 0):
                print(slices)
            for i in range(pred.shape[1]):
                for j in range(pred.shape[2]):
                    pred1[slices, i, j, 0] = np.argmax(pred[slices, i, j, :])
        print(pred1.shape)
        np.save('./fcnn_tfcv5_1.npy', pred1)


class FCNN2():
    def conv_block(self, x, nb_filters, kernel, depth, factor, acti):
        if depth > 0:
            m = Conv2D(nb_filters, kernel, padding='same')(x)
            m = BatchNormalization()(m)
            x = Activation(activation=acti)(m)

            x = Conv2D(nb_filters, kernel, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation(activation=acti)(x)
            x = MaxPool2D()(x)

            x = self.conv_block(x, int(nb_filters * factor), kernel, depth - 1, factor, acti)
            x = UpSampling2D()(x)
            x = Conv2D(nb_filters, kernel, activation=acti, padding='same')(x)
            # x = Conv2DTranspose(nb_filters, kernel, strides=2, activation=acti)(x)
            x = Concatenate(axis=-1)([x, m])
        x = Conv2D(nb_filters, kernel, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation=acti)(x)
        x = Conv2D(nb_filters, kernel, padding='same')(x)
        x = BatchNormalization()(x)
        return Activation(activation=acti)(x)

    def net(self, img_shape, n_out=4, dim=32, depth=4, factor=2, acti='relu', flatten=False):
        i = Input(shape=img_shape)
        o = self.conv_block(i, nb_filters=dim, kernel=3, depth=depth, factor=factor, acti=acti)
        o = Conv2D(n_out, (1, 1))(o)
        # if flatten:
        #     o = Reshape(n_out, img_shape[0] * img_shape[1])(o)
        #     o = Permute((2, 1))(o)
        o = Activation('softmax')(o)
        return Model(inputs=i, outputs=o)

    def step_decay(slef, epochs):
        init_rate = 0.001
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
        print("###数据增强前###", X_train.shape, Y_train.shape)

        if mode:
            X_tmp = rotate(X_train[0:7500], 90, (1, 2))  # 作旋转进行数据增强吗
            X_tmp = np.append(X_tmp, rotate(X_train[7500:], -90, (1, 2)), axis=0)
            X_train = np.append(X_train, X_tmp, axis=0)

            Y_tmp = rotate(Y_train[0:7500], 90, (1, 2))
            Y_tmp = np.append(Y_tmp, rotate(Y_train[7500:], -90, (1, 2)), axis=0)
            Y_train = np.append(Y_train, Y_tmp, axis=0)

            X_train, Y_train = shuffle(X_train, Y_train)
            print("###数据增强后###", X_train.shape, Y_train.shape)
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
        # y_true = y_true[:, :, :, 4:]
        # y_pred = y_pred[:, :, :, 4:]

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

        # y_true = Concatenate(axis=-1)([y_true[:, :, :, 1:2], y_true[:, :, :, 3:]])
        # y_pred = Concatenate(axis=-1)([y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]])
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
        # fcnn = FCNN()
        model = self.net((240, 240, 4), dim=32, factor=2)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        model.summary()

        change_lr = LearningRateScheduler(self.step_decay)
        sgd = SGD(lr=0.003, momentum=0.9, decay=0, nesterov=True)
        adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', self.dice_comp,
                                                                                  self.dice_core, self.dice_en])
        checkpointer = ModelCheckpoint(filepath='../weights/FCNN/2/deconv_150_5.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=70, batch_size=10,
                         shuffle=True, callbacks=[checkpointer, change_lr])
        with open('../weights/FCNN/2/fcnn_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)

    def test(self):
        model = self.net((240, 240, 4), dim=32, factor=2)
        model.load_weights('./FCNN2/weights/deconv_150_5.69-0.02.hdf5')
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        del X_train, Y_train, Y_test
        model.summary()
        print(X_test.shape)
        pred = model.predict(X_test, verbose=1)
        np.save('./FCNN2/fcnn_tfcv5_5.npy', pred)
        print(pred.shape)
        print('#### start test ###')
        pred1 = np.ndarray([pred.shape[0], pred.shape[1], pred.shape[2], 1], dtype=np.uint8)
        for slices in range(pred.shape[0]):
            if (slices % 155 == 0):
                print(slices)
            for i in range(pred.shape[1]):
                for j in range(pred.shape[2]):
                    pred1[slices, i, j, 0] = np.argmax(pred[slices, i, j, :])
        print(pred1.shape)
        np.save('./fcnn_tfcv5_1.npy', pred1)


class FCNN3():
    def conv_block(self, x, nb_filters, kernel, depth, factor, acti):
        if depth > 0:
            m = Conv2D(nb_filters, kernel, padding='same')(x)
            m = BatchNormalization()(m)
            x = Activation(activation=acti)(m)

            x = Conv2D(nb_filters, kernel, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation(activation=acti)(x)
            x = MaxPool2D()(x)

            x = self.conv_block(x, int(nb_filters * factor), kernel, depth - 1, factor, acti)
            x = UpSampling2D()(x)
            x = Conv2D(nb_filters, kernel, activation=acti, padding='same')(x)
            # x = Conv2DTranspose(nb_filters, kernel, strides=2, activation=acti)(x)
            x = Concatenate(axis=-1)([x, m])
        x = Conv2D(nb_filters, kernel, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation=acti)(x)
        x = Conv2D(nb_filters, kernel, padding='same')(x)
        x = BatchNormalization()(x)
        return Activation(activation=acti)(x)

    def net(self, img_shape, n_out=4, dim=32, depth=4, factor=2, acti='relu', flatten=False):
        i = Input(shape=img_shape)
        o = self.conv_block(i, nb_filters=dim, kernel=3, depth=depth, factor=factor, acti=acti)
        o = Conv2D(n_out, (1, 1))(o)
        # if flatten:
        #     o = Reshape(n_out, img_shape[0] * img_shape[1])(o)
        #     o = Permute((2, 1))(o)
        o = Activation('softmax')(o)
        return Model(inputs=i, outputs=o)

    def step_decay(slef, epochs):
        init_rate = 0.001
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
        print("###数据增强前###", X_train.shape, Y_train.shape)

        if mode:
            X_tmp = rotate(X_train[0:7500], 90, (1, 2))  # 作旋转进行数据增强吗
            X_tmp = np.append(X_tmp, rotate(X_train[7500:], -90, (1, 2)), axis=0)
            X_train = np.append(X_train, X_tmp, axis=0)

            Y_tmp = rotate(Y_train[0:7500], 90, (1, 2))
            Y_tmp = np.append(Y_tmp, rotate(Y_train[7500:], -90, (1, 2)), axis=0)
            Y_train = np.append(Y_train, Y_tmp, axis=0)

            X_train, Y_train = shuffle(X_train, Y_train)
            print("###数据增强后###", X_train.shape, Y_train.shape)
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
        # y_true = y_true[:, :, :, 4:]
        # y_pred = y_pred[:, :, :, 4:]

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

        # y_true = Concatenate(axis=-1)([y_true[:, :, :, 1:2], y_true[:, :, :, 3:]])
        # y_pred = Concatenate(axis=-1)([y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]])
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
        # fcnn = FCNN()
        model = self.net((240, 240, 4), dim=32, factor=2)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        model.summary()

        change_lr = LearningRateScheduler(self.step_decay)
        sgd = SGD(lr=0.003, momentum=0.9, decay=0, nesterov=True)
        adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', self.dice_comp,
                                                                                  self.dice_core, self.dice_en])
        checkpointer = ModelCheckpoint(filepath='../weights/FCNN/3/deconv_150_5.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=70, batch_size=10,
                         shuffle=True, callbacks=[checkpointer, change_lr])
        with open('../weights/FCNN/3/fcnn_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)

    def test(self):
        model = self.net((240, 240, 4), dim=32, factor=2)
        model.load_weights('./FCNN3/weights/deconv_150_5.69-0.02.hdf5')
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        del X_train, Y_train, Y_test
        model.summary()
        print(X_test.shape)
        pred = model.predict(X_test, verbose=1)
        np.save('./FCNN3/fcnn_tfcv5_5.npy', pred)
        print(pred.shape)
        print('#### start test ###')
        pred1 = np.ndarray([pred.shape[0], pred.shape[1], pred.shape[2], 1], dtype=np.uint8)
        for slices in range(pred.shape[0]):
            if (slices % 155 == 0):
                print(slices)
            for i in range(pred.shape[1]):
                for j in range(pred.shape[2]):
                    pred1[slices, i, j, 0] = np.argmax(pred[slices, i, j, :])
        print(pred1.shape)
        np.save('./fcnn_tfcv5_1.npy', pred1)


class FCNN4():
    def conv_block(self, x, nb_filters, kernel, depth, factor, acti):
        if depth > 0:
            m = Conv2D(nb_filters, kernel, padding='same')(x)
            m = BatchNormalization()(m)
            x = Activation(activation=acti)(m)

            x = Conv2D(nb_filters, kernel, padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation(activation=acti)(x)
            x = MaxPool2D()(x)

            x = self.conv_block(x, int(nb_filters * factor), kernel, depth - 1, factor, acti)
            x = UpSampling2D()(x)
            x = Conv2D(nb_filters, kernel, activation=acti, padding='same')(x)
            # x = Conv2DTranspose(nb_filters, kernel, strides=2, activation=acti)(x)
            x = Concatenate(axis=-1)([x, m])
        x = Conv2D(nb_filters, kernel, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation=acti)(x)
        x = Conv2D(nb_filters, kernel, padding='same')(x)
        x = BatchNormalization()(x)
        return Activation(activation=acti)(x)

    def net(self, img_shape, n_out=4, dim=32, depth=4, factor=2, acti='relu', flatten=False):
        i = Input(shape=img_shape)
        o = self.conv_block(i, nb_filters=dim, kernel=3, depth=depth, factor=factor, acti=acti)
        o = Conv2D(n_out, (1, 1))(o)
        # if flatten:
        #     o = Reshape(n_out, img_shape[0] * img_shape[1])(o)
        #     o = Permute((2, 1))(o)
        o = Activation('softmax')(o)
        return Model(inputs=i, outputs=o)

    def step_decay(slef, epochs):
        init_rate = 0.001
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
        print("###数据增强前###", X_train.shape, Y_train.shape)

        if mode:
            X_tmp = rotate(X_train[0:7500], 90, (1, 2))  # 作旋转进行数据增强吗
            X_tmp = np.append(X_tmp, rotate(X_train[7500:], -90, (1, 2)), axis=0)
            X_train = np.append(X_train, X_tmp, axis=0)

            Y_tmp = rotate(Y_train[0:7500], 90, (1, 2))
            Y_tmp = np.append(Y_tmp, rotate(Y_train[7500:], -90, (1, 2)), axis=0)
            Y_train = np.append(Y_train, Y_tmp, axis=0)

            X_train, Y_train = shuffle(X_train, Y_train)
            print("###数据增强后###", X_train.shape, Y_train.shape)
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
        # y_true = y_true[:, :, :, 4:]
        # y_pred = y_pred[:, :, :, 4:]

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

        # y_true = Concatenate(axis=-1)([y_true[:, :, :, 1:2], y_true[:, :, :, 3:]])
        # y_pred = Concatenate(axis=-1)([y_pred[:, :, :, 1:2], y_pred[:, :, :, 3:]])
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
        # fcnn = FCNN()
        model = self.net((240, 240, 4), dim=32, factor=2)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        model.summary()

        change_lr = LearningRateScheduler(self.step_decay)
        sgd = SGD(lr=0.003, momentum=0.9, decay=0, nesterov=True)
        adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', self.dice_comp,
                                                                                  self.dice_core, self.dice_en])
        checkpointer = ModelCheckpoint(filepath='../weights/FCNN/4/deconv_150_5.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=70, batch_size=10,
                         shuffle=True, callbacks=[checkpointer, change_lr])
        with open('../weights/FCNN/4/fcnn_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)

    def test(self):
        model = self.net((240, 240, 4), dim=32, factor=2)
        model.load_weights('./FCNN4/weights/deconv_150_5.69-0.02.hdf5')
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        del X_train, Y_train, Y_test
        model.summary()
        print(X_test.shape)
        pred = model.predict(X_test, verbose=1)
        np.save('./FCNN4/fcnn_tfcv5_5.npy', pred)
        print(pred.shape)
        print('#### start test ###')
        pred1 = np.ndarray([pred.shape[0], pred.shape[1], pred.shape[2], 1], dtype=np.uint8)
        for slices in range(pred.shape[0]):
            if (slices % 155 == 0):
                print(slices)
            for i in range(pred.shape[1]):
                for j in range(pred.shape[2]):
                    pred1[slices, i, j, 0] = np.argmax(pred[slices, i, j, :])
        print(pred1.shape)
        np.save('./fcnn_tfcv5_1.npy', pred1)

if __name__ == '__main__':
    # fcnn = FCNN()
    # fcnn.train()
    # fcnn.train()
    FCNN0 = FCNN0()
    FCNN0.train()

    FCNN1 = FCNN1()
    FCNN1.train()

    FCNN2 = FCNN2()
    FCNN2.train()

    FCNN3 = FCNN3()
    FCNN3.train()

    FCNN4 = FCNN4()
    FCNN4.train()

