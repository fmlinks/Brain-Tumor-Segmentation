import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler

from keras.layers import *
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import *
import numpy as np
import json
from sklearn.utils import shuffle
from scipy.ndimage.interpolation import rotate
import tensorflow as tf
from keras.applications import InceptionV3

np.random.seed(1234)

# model = InceptionV3(include_top=False, input_shape=(160, 160, 3))
# model.summary()


from keras.callbacks import TensorBoard


class FPN0:

    def __init__(self, inputshape=(240, 240, 4), classes=4):
        self.input = Input(inputshape)
        self.classes = classes

    def get_model(self):

        # 160 160 4
        conv1_1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv1_1')(self.input)
        bn1_1 = BatchNormalization()(conv1_1)
        acti1_1 = Activation(activation='relu')(bn1_1)

        conv1_2 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv1_2')(acti1_1)
        bn1_2 = BatchNormalization()(conv1_2)
        acti1_2 = Activation(activation='relu')(bn1_2)
        maxpool1 = MaxPool2D()(acti1_2)

        conv2_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv2_1')(maxpool1)
        bn2_1 = BatchNormalization()(conv2_1)
        acti2_1 = Activation(activation='relu')(bn2_1)

        conv2_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv2_2')(acti2_1)
        bn2_2 = BatchNormalization()(conv2_2)
        acti2_2 = Activation(activation='relu')(bn2_2)
        maxpool2 = MaxPool2D()(acti2_2)

        conv3_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv3_1')(maxpool2)
        bn3_1 = BatchNormalization()(conv3_1)
        acti3_1 = Activation(activation='relu')(bn3_1)

        conv3_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv3_2')(acti3_1)
        bn3_2 = BatchNormalization()(conv3_2)
        acti3_2 = Activation(activation='relu')(bn3_2)
        maxpool3 = MaxPool2D()(acti3_2)

        conv4_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv4_1')(maxpool3)
        bn4_1 = BatchNormalization()(conv4_1)
        acti4_1 = Activation(activation='relu')(bn4_1)

        conv4_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv4_2')(acti4_1)
        bn4_2 = BatchNormalization()(conv4_2)
        acti4_2 = Activation(activation='relu')(bn4_2)
        maxpool4 = MaxPool2D()(acti4_2)

        conv5_1 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv5_1')(maxpool4)
        bn5_1 = BatchNormalization()(conv5_1)
        acti5_1 = Activation(activation='relu')(bn5_1)

        conv5_2 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv5_2')(acti5_1)
        bn5_2 = BatchNormalization()(conv5_2)
        acti5_2 = Activation(activation='relu')(bn5_2)

        up6 = Conv2D(filters=256, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up6')(UpSampling2D()(acti5_2))

        concat6 = Concatenate(axis=-1)([up6, acti4_2])

        conv6_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv6_1')(concat6)
        bn6_1 = BatchNormalization()(conv6_1)
        acti6_1 = Activation(activation='relu')(bn6_1)

        conv6_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv6_2')(acti6_1)
        bn6_2 = BatchNormalization()(conv6_2)
        acti6_2 = Activation(activation='relu')(bn6_2)

        up7 = Conv2D(filters=128, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up7')(UpSampling2D()(acti6_2))

        concat7 = Concatenate(axis=-1)([up7, acti3_2])

        conv7_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv7_1')(concat7)
        bn7_1 = BatchNormalization()(conv7_1)
        acti7_1 = Activation(activation='relu')(bn7_1)

        conv7_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv7_2')(acti7_1)
        bn7_2 = BatchNormalization()(conv7_2)
        acti7_2 = Activation(activation='relu')(bn7_2)

        up8 = Conv2D(filters=64, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up8')(UpSampling2D()(acti7_2))

        concat8 = Concatenate(axis=-1)([up8, acti2_2])

        conv8_1 = Conv2D(filters=64, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv8_1')(concat8)
        bn8_1 = BatchNormalization()(conv8_1)
        acti8_1 = Activation(activation='relu')(bn8_1)

        conv8_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv8_2')(acti8_1)
        bn8_2 = BatchNormalization()(conv8_2)
        acti8_2 = Activation(activation='relu')(bn8_2)

        up9 = Conv2D(filters=32, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up9')(UpSampling2D()(acti8_2))

        concat9 = Concatenate(axis=-1)([up9, acti1_2])

        conv9_1 = Conv2D(filters=32, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv9_1')(concat9)
        bn9_1 = BatchNormalization()(conv9_1)
        acti9_1 = Activation(activation='relu')(bn9_1)

        conv9_2 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv9_2')(acti9_1)
        bn9_2 = BatchNormalization()(conv9_2)
        acti9_2 = Activation(activation='relu')(bn9_2)

        # ==================================nconv2=====================================#
        down2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal',
                       name='down2')(acti9_2)
        bn_down2 = BatchNormalization()(down2)
        acti_down2 = Activation(activation='relu')(bn_down2)

        concat2 = Concatenate(axis=-1)([acti_down2, acti8_2])

        nconv2_1 = Conv2D(filters=64, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv2_1')(concat2)
        bn_nconv2_1 = BatchNormalization()(nconv2_1)
        acti_nconv2_1 = Activation(activation='relu')(bn_nconv2_1)

        nconv2_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv2_2')(acti_nconv2_1)
        bn_nconv2_2 = BatchNormalization()(nconv2_2)
        acti_nconv2_2 = Activation(activation='relu')(bn_nconv2_2)

        # ###====== Deconv + Conv ======####

        # =================================nconv3=======================================#

        down3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal',
                       name='down3')(acti_nconv2_2)
        bn_down3 = BatchNormalization()(down3)
        acti_down3 = Activation(activation='relu')(bn_down3)

        concat3 = Concatenate(axis=-1)([acti_down3, acti7_2])

        nconv3_1 = Conv2D(filters=128, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv3_1')(concat3)
        bn_nconv3_1 = BatchNormalization()(nconv3_1)
        acti_nconv3_1 = Activation(activation='relu')(bn_nconv3_1)

        nconv3_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv3_2')(acti_nconv3_1)
        bn_nconv3_2 = BatchNormalization()(nconv3_2)
        acti_nconv3_2 = Activation(activation='relu')(bn_nconv3_2)

        # ###====== Deconv + Conv ======####
        ####################################

        # =================================nconv4=======================================#

        down4 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal',
                       name='down4')(acti_nconv3_2)
        bn_down4 = BatchNormalization()(down4)
        acti_down4 = Activation(activation='relu')(bn_down4)

        concat4 = Concatenate(axis=-1)([acti_down4, acti6_2])

        nconv4_1 = Conv2D(filters=256, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv4_1')(concat4)
        bn_nconv4_1 = BatchNormalization()(nconv4_1)
        acti_nconv4_1 = Activation(activation='relu')(bn_nconv4_1)

        nconv4_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv4_2')(acti_nconv4_1)
        bn_nconv4_2 = BatchNormalization()(nconv4_2)
        acti_nconv4_2 = Activation(activation='relu')(bn_nconv4_2)

        # ###====== Deconv + Conv ======####
        ####################################

        acti_nconv2_up = UpSampling2D(size=(2, 2))(acti_nconv2_2)
        nconv2_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                           name='nconv2_up')(acti_nconv2_up)
        nconv2_up = BatchNormalization()(nconv2_up)
        nconv2_up = Activation(activation='relu')(nconv2_up)

        acti_nconv3_up = UpSampling2D(size=(4, 4))(acti_nconv3_2)
        nconv3_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                           name='nconv3_up')(acti_nconv3_up)
        nconv3_up = BatchNormalization()(nconv3_up)
        nconv3_up = Activation(activation='relu')(nconv3_up)

        acti_nconv4_up = UpSampling2D(size=(8, 8))(acti_nconv4_2)
        nconv4_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                           name='nconv4_up')(acti_nconv4_up)
        nconv4_up = BatchNormalization()(nconv4_up)
        nconv4_up = Activation(activation='relu')(nconv4_up)

        conv9_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                          name='conv9_up')(acti9_2)
        conv9_up = BatchNormalization()(conv9_up)
        conv9_up = Activation(activation='relu')(conv9_up)

        conv10 = Add()([nconv2_up, nconv3_up, nconv4_up, conv9_up])
        output = Conv2D(filters=self.classes, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                        name='output', activation='softmax')(conv10)

        # conv10 = Conv2D(filters=self.classes, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
        #                 name='conv10')(acti9_2)
        #

        model = Model(inputs=self.input, outputs=output)

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
        print("###数据增强前###", X_train.shape, Y_train.shape)

        if mode:
            X_tmp = rotate(X_train[0:6000], 90, (1, 2))  # 作旋转进行数据增强吗
            X_tmp = np.append(X_tmp, rotate(X_train[6000:], -90, (1, 2)), axis=0)
            X_train = np.append(X_train, X_tmp, axis=0)

            Y_tmp = rotate(Y_train[0:6000], 90, (1, 2))
            Y_tmp = np.append(Y_tmp, rotate(Y_train[6000:], -90, (1, 2)), axis=0)
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
        model = self.get_model()
        model.summary()
        # plot_model(model, to_file='./DUnet.png', show_shapes=True, show_layer_names=True)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)

        change_lr = LearningRateScheduler(self.step_decay)
        sgd = SGD(lr=0.003, momentum=0.9, decay=0, nesterov=True)
        adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', self.dice_comp,
                                                                                  self.dice_core, self.dice_en])
        checkpointer = ModelCheckpoint(filepath='../weights/PA+FP/0/deconv_150_5_test.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=70, batch_size=8,
                         shuffle=True, callbacks=[checkpointer, change_lr])
        with open('../weights/PA+FP/0/PA+FP_tfcv5_test.json', 'w') as f:
            json.dump(hist.history, f)

        # model.fit(X_train, Y_train,
        #           validation_data=(X_test, Y_test),
        #           epochs=70,
        #           batch_size=8,
        #           shuffle=True,
        #           callbacks=[TensorBoard(log_dir='./log'), checkpointer])

    def test(self):
        model = self.conv_block()
        model.summary()
        model.load_weights('./PA+FP0/weights/dunet_150_5.69-0.03.hdf5')
        X_train, Y_train, X_test, Y_test = self.load_data('./data/BraTS15_all/', mode=False)
        del X_train, Y_train, Y_test
        model.summary()
        print(X_test.shape)
        pred = model.predict(X_test, verbose=1)
        np.save('./dunet_tfcv5_5.npy', pred)
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
        np.save('./dunet_tfcv5_1.npy', pred1)

class FPN1:

    def __init__(self, inputshape=(160, 160, 4), classes=4):
        self.input = Input(inputshape)
        self.classes = classes

    def get_model(self):

        # 160 160 4
        conv1_1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv1_1')(self.input)
        bn1_1 = BatchNormalization()(conv1_1)
        acti1_1 = Activation(activation='relu')(bn1_1)

        conv1_2 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv1_2')(acti1_1)
        bn1_2 = BatchNormalization()(conv1_2)
        acti1_2 = Activation(activation='relu')(bn1_2)
        maxpool1 = MaxPool2D()(acti1_2)

        conv2_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv2_1')(maxpool1)
        bn2_1 = BatchNormalization()(conv2_1)
        acti2_1 = Activation(activation='relu')(bn2_1)

        conv2_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv2_2')(acti2_1)
        bn2_2 = BatchNormalization()(conv2_2)
        acti2_2 = Activation(activation='relu')(bn2_2)
        maxpool2 = MaxPool2D()(acti2_2)

        conv3_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv3_1')(maxpool2)
        bn3_1 = BatchNormalization()(conv3_1)
        acti3_1 = Activation(activation='relu')(bn3_1)

        conv3_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv3_2')(acti3_1)
        bn3_2 = BatchNormalization()(conv3_2)
        acti3_2 = Activation(activation='relu')(bn3_2)
        maxpool3 = MaxPool2D()(acti3_2)

        conv4_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv4_1')(maxpool3)
        bn4_1 = BatchNormalization()(conv4_1)
        acti4_1 = Activation(activation='relu')(bn4_1)

        conv4_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv4_2')(acti4_1)
        bn4_2 = BatchNormalization()(conv4_2)
        acti4_2 = Activation(activation='relu')(bn4_2)
        maxpool4 = MaxPool2D()(acti4_2)

        conv5_1 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv5_1')(maxpool4)
        bn5_1 = BatchNormalization()(conv5_1)
        acti5_1 = Activation(activation='relu')(bn5_1)

        conv5_2 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv5_2')(acti5_1)
        bn5_2 = BatchNormalization()(conv5_2)
        acti5_2 = Activation(activation='relu')(bn5_2)

        up6 = Conv2D(filters=256, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up6')(UpSampling2D()(acti5_2))

        concat6 = Concatenate(axis=-1)([up6, acti4_2])

        conv6_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv6_1')(concat6)
        bn6_1 = BatchNormalization()(conv6_1)
        acti6_1 = Activation(activation='relu')(bn6_1)

        conv6_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv6_2')(acti6_1)
        bn6_2 = BatchNormalization()(conv6_2)
        acti6_2 = Activation(activation='relu')(bn6_2)

        up7 = Conv2D(filters=128, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up7')(UpSampling2D()(acti6_2))

        concat7 = Concatenate(axis=-1)([up7, acti3_2])

        conv7_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv7_1')(concat7)
        bn7_1 = BatchNormalization()(conv7_1)
        acti7_1 = Activation(activation='relu')(bn7_1)

        conv7_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv7_2')(acti7_1)
        bn7_2 = BatchNormalization()(conv7_2)
        acti7_2 = Activation(activation='relu')(bn7_2)

        up8 = Conv2D(filters=64, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up8')(UpSampling2D()(acti7_2))

        concat8 = Concatenate(axis=-1)([up8, acti2_2])

        conv8_1 = Conv2D(filters=64, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv8_1')(concat8)
        bn8_1 = BatchNormalization()(conv8_1)
        acti8_1 = Activation(activation='relu')(bn8_1)

        conv8_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv8_2')(acti8_1)
        bn8_2 = BatchNormalization()(conv8_2)
        acti8_2 = Activation(activation='relu')(bn8_2)

        up9 = Conv2D(filters=32, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up9')(UpSampling2D()(acti8_2))

        concat9 = Concatenate(axis=-1)([up9, acti1_2])

        conv9_1 = Conv2D(filters=32, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv9_1')(concat9)
        bn9_1 = BatchNormalization()(conv9_1)
        acti9_1 = Activation(activation='relu')(bn9_1)

        conv9_2 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv9_2')(acti9_1)
        bn9_2 = BatchNormalization()(conv9_2)
        acti9_2 = Activation(activation='relu')(bn9_2)

        # ==================================nconv2=====================================#
        down2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal',
                       name='down2')(acti9_2)
        bn_down2 = BatchNormalization()(down2)
        acti_down2 = Activation(activation='relu')(bn_down2)

        concat2 = Concatenate(axis=-1)([acti_down2, acti8_2])

        nconv2_1 = Conv2D(filters=64, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv2_1')(concat2)
        bn_nconv2_1 = BatchNormalization()(nconv2_1)
        acti_nconv2_1 = Activation(activation='relu')(bn_nconv2_1)

        nconv2_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv2_2')(acti_nconv2_1)
        bn_nconv2_2 = BatchNormalization()(nconv2_2)
        acti_nconv2_2 = Activation(activation='relu')(bn_nconv2_2)

        # ###====== Deconv + Conv ======####

        # =================================nconv3=======================================#

        down3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal',
                       name='down3')(acti_nconv2_2)
        bn_down3 = BatchNormalization()(down3)
        acti_down3 = Activation(activation='relu')(bn_down3)

        concat3 = Concatenate(axis=-1)([acti_down3, acti7_2])

        nconv3_1 = Conv2D(filters=128, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv3_1')(concat3)
        bn_nconv3_1 = BatchNormalization()(nconv3_1)
        acti_nconv3_1 = Activation(activation='relu')(bn_nconv3_1)

        nconv3_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv3_2')(acti_nconv3_1)
        bn_nconv3_2 = BatchNormalization()(nconv3_2)
        acti_nconv3_2 = Activation(activation='relu')(bn_nconv3_2)

        # ###====== Deconv + Conv ======####
        ####################################

        # =================================nconv4=======================================#

        down4 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal',
                       name='down4')(acti_nconv3_2)
        bn_down4 = BatchNormalization()(down4)
        acti_down4 = Activation(activation='relu')(bn_down4)

        concat4 = Concatenate(axis=-1)([acti_down4, acti6_2])

        nconv4_1 = Conv2D(filters=256, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv4_1')(concat4)
        bn_nconv4_1 = BatchNormalization()(nconv4_1)
        acti_nconv4_1 = Activation(activation='relu')(bn_nconv4_1)

        nconv4_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv4_2')(acti_nconv4_1)
        bn_nconv4_2 = BatchNormalization()(nconv4_2)
        acti_nconv4_2 = Activation(activation='relu')(bn_nconv4_2)

        # ###====== Deconv + Conv ======####
        ####################################

        acti_nconv2_up = UpSampling2D(size=(2, 2))(acti_nconv2_2)
        nconv2_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                           name='nconv2_up')(acti_nconv2_up)
        nconv2_up = BatchNormalization()(nconv2_up)
        nconv2_up = Activation(activation='relu')(nconv2_up)

        acti_nconv3_up = UpSampling2D(size=(4, 4))(acti_nconv3_2)
        nconv3_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                           name='nconv3_up')(acti_nconv3_up)
        nconv3_up = BatchNormalization()(nconv3_up)
        nconv3_up = Activation(activation='relu')(nconv3_up)

        acti_nconv4_up = UpSampling2D(size=(8, 8))(acti_nconv4_2)
        nconv4_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                           name='nconv4_up')(acti_nconv4_up)
        nconv4_up = BatchNormalization()(nconv4_up)
        nconv4_up = Activation(activation='relu')(nconv4_up)

        conv9_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                          name='conv9_up')(acti9_2)
        conv9_up = BatchNormalization()(conv9_up)
        conv9_up = Activation(activation='relu')(conv9_up)

        conv10 = Add()([nconv2_up, nconv3_up, nconv4_up, conv9_up])
        output = Conv2D(filters=self.classes, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                        name='output', activation='softmax')(conv10)

        # conv10 = Conv2D(filters=self.classes, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
        #                 name='conv10')(acti9_2)
        #

        model = Model(inputs=self.input, outputs=output)

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
        print("###数据增强前###", X_train.shape, Y_train.shape)

        if mode:
            X_tmp = rotate(X_train[0:6000], 90, (1, 2))  # 作旋转进行数据增强吗
            X_tmp = np.append(X_tmp, rotate(X_train[6000:], -90, (1, 2)), axis=0)
            X_train = np.append(X_train, X_tmp, axis=0)

            Y_tmp = rotate(Y_train[0:6000], 90, (1, 2))
            Y_tmp = np.append(Y_tmp, rotate(Y_train[6000:], -90, (1, 2)), axis=0)
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
        model = self.get_model()
        model.summary()
        # plot_model(model, to_file='./DUnet.png', show_shapes=True, show_layer_names=True)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)

        change_lr = LearningRateScheduler(self.step_decay)
        sgd = SGD(lr=0.003, momentum=0.9, decay=0, nesterov=True)
        adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', self.dice_comp,
                                                                                  self.dice_core, self.dice_en])
        checkpointer = ModelCheckpoint(filepath='../weights/PA+FP/1/deconv_150_5.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=70, batch_size=16,
                         shuffle=True, callbacks=[checkpointer, change_lr])
        with open('../weights/PA+FP/1/PA+FP_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)

        # model.fit(X_train, Y_train,
        #           validation_data=(X_test, Y_test),
        #           epochs=70,
        #           batch_size=8,
        #           shuffle=True,
        #           callbacks=[TensorBoard(log_dir='./log'), checkpointer])

    def test(self):
        model = self.conv_block()
        model.summary()
        model.load_weights('./PA+FP1/weights/dunet_150_5.69-0.03.hdf5')
        X_train, Y_train, X_test, Y_test = self.load_data('./data/BraTS15_all/', mode=False)
        del X_train, Y_train, Y_test
        model.summary()
        print(X_test.shape)
        pred = model.predict(X_test, verbose=1)
        np.save('./dunet_tfcv5_5.npy', pred)
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
        np.save('./dunet_tfcv5_1.npy', pred1)

class FPN2:

    def __init__(self, inputshape=(160, 160, 4), classes=4):
        self.input = Input(inputshape)
        self.classes = classes

    def get_model(self):

        # 160 160 4
        conv1_1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv1_1')(self.input)
        bn1_1 = BatchNormalization()(conv1_1)
        acti1_1 = Activation(activation='relu')(bn1_1)

        conv1_2 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv1_2')(acti1_1)
        bn1_2 = BatchNormalization()(conv1_2)
        acti1_2 = Activation(activation='relu')(bn1_2)
        maxpool1 = MaxPool2D()(acti1_2)

        conv2_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv2_1')(maxpool1)
        bn2_1 = BatchNormalization()(conv2_1)
        acti2_1 = Activation(activation='relu')(bn2_1)

        conv2_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv2_2')(acti2_1)
        bn2_2 = BatchNormalization()(conv2_2)
        acti2_2 = Activation(activation='relu')(bn2_2)
        maxpool2 = MaxPool2D()(acti2_2)

        conv3_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv3_1')(maxpool2)
        bn3_1 = BatchNormalization()(conv3_1)
        acti3_1 = Activation(activation='relu')(bn3_1)

        conv3_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv3_2')(acti3_1)
        bn3_2 = BatchNormalization()(conv3_2)
        acti3_2 = Activation(activation='relu')(bn3_2)
        maxpool3 = MaxPool2D()(acti3_2)

        conv4_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv4_1')(maxpool3)
        bn4_1 = BatchNormalization()(conv4_1)
        acti4_1 = Activation(activation='relu')(bn4_1)

        conv4_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv4_2')(acti4_1)
        bn4_2 = BatchNormalization()(conv4_2)
        acti4_2 = Activation(activation='relu')(bn4_2)
        maxpool4 = MaxPool2D()(acti4_2)

        conv5_1 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv5_1')(maxpool4)
        bn5_1 = BatchNormalization()(conv5_1)
        acti5_1 = Activation(activation='relu')(bn5_1)

        conv5_2 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv5_2')(acti5_1)
        bn5_2 = BatchNormalization()(conv5_2)
        acti5_2 = Activation(activation='relu')(bn5_2)

        up6 = Conv2D(filters=256, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up6')(UpSampling2D()(acti5_2))

        concat6 = Concatenate(axis=-1)([up6, acti4_2])

        conv6_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv6_1')(concat6)
        bn6_1 = BatchNormalization()(conv6_1)
        acti6_1 = Activation(activation='relu')(bn6_1)

        conv6_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv6_2')(acti6_1)
        bn6_2 = BatchNormalization()(conv6_2)
        acti6_2 = Activation(activation='relu')(bn6_2)

        up7 = Conv2D(filters=128, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up7')(UpSampling2D()(acti6_2))

        concat7 = Concatenate(axis=-1)([up7, acti3_2])

        conv7_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv7_1')(concat7)
        bn7_1 = BatchNormalization()(conv7_1)
        acti7_1 = Activation(activation='relu')(bn7_1)

        conv7_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv7_2')(acti7_1)
        bn7_2 = BatchNormalization()(conv7_2)
        acti7_2 = Activation(activation='relu')(bn7_2)

        up8 = Conv2D(filters=64, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up8')(UpSampling2D()(acti7_2))

        concat8 = Concatenate(axis=-1)([up8, acti2_2])

        conv8_1 = Conv2D(filters=64, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv8_1')(concat8)
        bn8_1 = BatchNormalization()(conv8_1)
        acti8_1 = Activation(activation='relu')(bn8_1)

        conv8_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv8_2')(acti8_1)
        bn8_2 = BatchNormalization()(conv8_2)
        acti8_2 = Activation(activation='relu')(bn8_2)

        up9 = Conv2D(filters=32, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up9')(UpSampling2D()(acti8_2))

        concat9 = Concatenate(axis=-1)([up9, acti1_2])

        conv9_1 = Conv2D(filters=32, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv9_1')(concat9)
        bn9_1 = BatchNormalization()(conv9_1)
        acti9_1 = Activation(activation='relu')(bn9_1)

        conv9_2 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv9_2')(acti9_1)
        bn9_2 = BatchNormalization()(conv9_2)
        acti9_2 = Activation(activation='relu')(bn9_2)

        # ==================================nconv2=====================================#
        down2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal',
                       name='down2')(acti9_2)
        bn_down2 = BatchNormalization()(down2)
        acti_down2 = Activation(activation='relu')(bn_down2)

        concat2 = Concatenate(axis=-1)([acti_down2, acti8_2])

        nconv2_1 = Conv2D(filters=64, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv2_1')(concat2)
        bn_nconv2_1 = BatchNormalization()(nconv2_1)
        acti_nconv2_1 = Activation(activation='relu')(bn_nconv2_1)

        nconv2_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv2_2')(acti_nconv2_1)
        bn_nconv2_2 = BatchNormalization()(nconv2_2)
        acti_nconv2_2 = Activation(activation='relu')(bn_nconv2_2)

        # ###====== Deconv + Conv ======####

        # =================================nconv3=======================================#

        down3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal',
                       name='down3')(acti_nconv2_2)
        bn_down3 = BatchNormalization()(down3)
        acti_down3 = Activation(activation='relu')(bn_down3)

        concat3 = Concatenate(axis=-1)([acti_down3, acti7_2])

        nconv3_1 = Conv2D(filters=128, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv3_1')(concat3)
        bn_nconv3_1 = BatchNormalization()(nconv3_1)
        acti_nconv3_1 = Activation(activation='relu')(bn_nconv3_1)

        nconv3_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv3_2')(acti_nconv3_1)
        bn_nconv3_2 = BatchNormalization()(nconv3_2)
        acti_nconv3_2 = Activation(activation='relu')(bn_nconv3_2)

        # ###====== Deconv + Conv ======####
        ####################################

        # =================================nconv4=======================================#

        down4 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal',
                       name='down4')(acti_nconv3_2)
        bn_down4 = BatchNormalization()(down4)
        acti_down4 = Activation(activation='relu')(bn_down4)

        concat4 = Concatenate(axis=-1)([acti_down4, acti6_2])

        nconv4_1 = Conv2D(filters=256, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv4_1')(concat4)
        bn_nconv4_1 = BatchNormalization()(nconv4_1)
        acti_nconv4_1 = Activation(activation='relu')(bn_nconv4_1)

        nconv4_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv4_2')(acti_nconv4_1)
        bn_nconv4_2 = BatchNormalization()(nconv4_2)
        acti_nconv4_2 = Activation(activation='relu')(bn_nconv4_2)

        # ###====== Deconv + Conv ======####
        ####################################

        acti_nconv2_up = UpSampling2D(size=(2, 2))(acti_nconv2_2)
        nconv2_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                           name='nconv2_up')(acti_nconv2_up)
        nconv2_up = BatchNormalization()(nconv2_up)
        nconv2_up = Activation(activation='relu')(nconv2_up)

        acti_nconv3_up = UpSampling2D(size=(4, 4))(acti_nconv3_2)
        nconv3_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                           name='nconv3_up')(acti_nconv3_up)
        nconv3_up = BatchNormalization()(nconv3_up)
        nconv3_up = Activation(activation='relu')(nconv3_up)

        acti_nconv4_up = UpSampling2D(size=(8, 8))(acti_nconv4_2)
        nconv4_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                           name='nconv4_up')(acti_nconv4_up)
        nconv4_up = BatchNormalization()(nconv4_up)
        nconv4_up = Activation(activation='relu')(nconv4_up)

        conv9_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                          name='conv9_up')(acti9_2)
        conv9_up = BatchNormalization()(conv9_up)
        conv9_up = Activation(activation='relu')(conv9_up)

        conv10 = Add()([nconv2_up, nconv3_up, nconv4_up, conv9_up])
        output = Conv2D(filters=self.classes, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                        name='output', activation='softmax')(conv10)

        # conv10 = Conv2D(filters=self.classes, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
        #                 name='conv10')(acti9_2)
        #

        model = Model(inputs=self.input, outputs=output)

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
        print("###数据增强前###", X_train.shape, Y_train.shape)

        if mode:
            X_tmp = rotate(X_train[0:6000], 90, (1, 2))  # 作旋转进行数据增强吗
            X_tmp = np.append(X_tmp, rotate(X_train[6000:], -90, (1, 2)), axis=0)
            X_train = np.append(X_train, X_tmp, axis=0)

            Y_tmp = rotate(Y_train[0:6000], 90, (1, 2))
            Y_tmp = np.append(Y_tmp, rotate(Y_train[6000:], -90, (1, 2)), axis=0)
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
        model = self.get_model()
        model.summary()
        # plot_model(model, to_file='./DUnet.png', show_shapes=True, show_layer_names=True)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)

        change_lr = LearningRateScheduler(self.step_decay)
        sgd = SGD(lr=0.003, momentum=0.9, decay=0, nesterov=True)
        adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', self.dice_comp,
                                                                                  self.dice_core, self.dice_en])
        checkpointer = ModelCheckpoint(filepath='../weights/PA+FP/2/deconv_150_5.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=70, batch_size=16,
                         shuffle=True, callbacks=[checkpointer, change_lr])
        with open('../weights/PA+FP/2/PA+FP_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)

        # model.fit(X_train, Y_train,
        #           validation_data=(X_test, Y_test),
        #           epochs=70,
        #           batch_size=8,
        #           shuffle=True,
        #           callbacks=[TensorBoard(log_dir='./log'), checkpointer])

    def test(self):
        model = self.conv_block()
        model.summary()
        model.load_weights('./PA+FP2/weights/dunet_150_5.69-0.03.hdf5')
        X_train, Y_train, X_test, Y_test = self.load_data('./data/BraTS15_all/', mode=False)
        del X_train, Y_train, Y_test
        model.summary()
        print(X_test.shape)
        pred = model.predict(X_test, verbose=1)
        np.save('./dunet_tfcv5_5.npy', pred)
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
        np.save('./dunet_tfcv5_1.npy', pred1)

class FPN3:

    def __init__(self, inputshape=(160, 160, 4), classes=4):
        self.input = Input(inputshape)
        self.classes = classes

    def get_model(self):

        # 160 160 4
        conv1_1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv1_1')(self.input)
        bn1_1 = BatchNormalization()(conv1_1)
        acti1_1 = Activation(activation='relu')(bn1_1)

        conv1_2 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv1_2')(acti1_1)
        bn1_2 = BatchNormalization()(conv1_2)
        acti1_2 = Activation(activation='relu')(bn1_2)
        maxpool1 = MaxPool2D()(acti1_2)

        conv2_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv2_1')(maxpool1)
        bn2_1 = BatchNormalization()(conv2_1)
        acti2_1 = Activation(activation='relu')(bn2_1)

        conv2_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv2_2')(acti2_1)
        bn2_2 = BatchNormalization()(conv2_2)
        acti2_2 = Activation(activation='relu')(bn2_2)
        maxpool2 = MaxPool2D()(acti2_2)

        conv3_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv3_1')(maxpool2)
        bn3_1 = BatchNormalization()(conv3_1)
        acti3_1 = Activation(activation='relu')(bn3_1)

        conv3_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv3_2')(acti3_1)
        bn3_2 = BatchNormalization()(conv3_2)
        acti3_2 = Activation(activation='relu')(bn3_2)
        maxpool3 = MaxPool2D()(acti3_2)

        conv4_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv4_1')(maxpool3)
        bn4_1 = BatchNormalization()(conv4_1)
        acti4_1 = Activation(activation='relu')(bn4_1)

        conv4_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv4_2')(acti4_1)
        bn4_2 = BatchNormalization()(conv4_2)
        acti4_2 = Activation(activation='relu')(bn4_2)
        maxpool4 = MaxPool2D()(acti4_2)

        conv5_1 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv5_1')(maxpool4)
        bn5_1 = BatchNormalization()(conv5_1)
        acti5_1 = Activation(activation='relu')(bn5_1)

        conv5_2 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv5_2')(acti5_1)
        bn5_2 = BatchNormalization()(conv5_2)
        acti5_2 = Activation(activation='relu')(bn5_2)

        up6 = Conv2D(filters=256, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up6')(UpSampling2D()(acti5_2))

        concat6 = Concatenate(axis=-1)([up6, acti4_2])

        conv6_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv6_1')(concat6)
        bn6_1 = BatchNormalization()(conv6_1)
        acti6_1 = Activation(activation='relu')(bn6_1)

        conv6_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv6_2')(acti6_1)
        bn6_2 = BatchNormalization()(conv6_2)
        acti6_2 = Activation(activation='relu')(bn6_2)

        up7 = Conv2D(filters=128, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up7')(UpSampling2D()(acti6_2))

        concat7 = Concatenate(axis=-1)([up7, acti3_2])

        conv7_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv7_1')(concat7)
        bn7_1 = BatchNormalization()(conv7_1)
        acti7_1 = Activation(activation='relu')(bn7_1)

        conv7_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv7_2')(acti7_1)
        bn7_2 = BatchNormalization()(conv7_2)
        acti7_2 = Activation(activation='relu')(bn7_2)

        up8 = Conv2D(filters=64, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up8')(UpSampling2D()(acti7_2))

        concat8 = Concatenate(axis=-1)([up8, acti2_2])

        conv8_1 = Conv2D(filters=64, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv8_1')(concat8)
        bn8_1 = BatchNormalization()(conv8_1)
        acti8_1 = Activation(activation='relu')(bn8_1)

        conv8_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv8_2')(acti8_1)
        bn8_2 = BatchNormalization()(conv8_2)
        acti8_2 = Activation(activation='relu')(bn8_2)

        up9 = Conv2D(filters=32, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up9')(UpSampling2D()(acti8_2))

        concat9 = Concatenate(axis=-1)([up9, acti1_2])

        conv9_1 = Conv2D(filters=32, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv9_1')(concat9)
        bn9_1 = BatchNormalization()(conv9_1)
        acti9_1 = Activation(activation='relu')(bn9_1)

        conv9_2 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv9_2')(acti9_1)
        bn9_2 = BatchNormalization()(conv9_2)
        acti9_2 = Activation(activation='relu')(bn9_2)

        # ==================================nconv2=====================================#
        down2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal',
                       name='down2')(acti9_2)
        bn_down2 = BatchNormalization()(down2)
        acti_down2 = Activation(activation='relu')(bn_down2)

        concat2 = Concatenate(axis=-1)([acti_down2, acti8_2])

        nconv2_1 = Conv2D(filters=64, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv2_1')(concat2)
        bn_nconv2_1 = BatchNormalization()(nconv2_1)
        acti_nconv2_1 = Activation(activation='relu')(bn_nconv2_1)

        nconv2_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv2_2')(acti_nconv2_1)
        bn_nconv2_2 = BatchNormalization()(nconv2_2)
        acti_nconv2_2 = Activation(activation='relu')(bn_nconv2_2)

        # ###====== Deconv + Conv ======####

        # =================================nconv3=======================================#

        down3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal',
                       name='down3')(acti_nconv2_2)
        bn_down3 = BatchNormalization()(down3)
        acti_down3 = Activation(activation='relu')(bn_down3)

        concat3 = Concatenate(axis=-1)([acti_down3, acti7_2])

        nconv3_1 = Conv2D(filters=128, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv3_1')(concat3)
        bn_nconv3_1 = BatchNormalization()(nconv3_1)
        acti_nconv3_1 = Activation(activation='relu')(bn_nconv3_1)

        nconv3_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv3_2')(acti_nconv3_1)
        bn_nconv3_2 = BatchNormalization()(nconv3_2)
        acti_nconv3_2 = Activation(activation='relu')(bn_nconv3_2)

        # ###====== Deconv + Conv ======####
        ####################################

        # =================================nconv4=======================================#

        down4 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal',
                       name='down4')(acti_nconv3_2)
        bn_down4 = BatchNormalization()(down4)
        acti_down4 = Activation(activation='relu')(bn_down4)

        concat4 = Concatenate(axis=-1)([acti_down4, acti6_2])

        nconv4_1 = Conv2D(filters=256, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv4_1')(concat4)
        bn_nconv4_1 = BatchNormalization()(nconv4_1)
        acti_nconv4_1 = Activation(activation='relu')(bn_nconv4_1)

        nconv4_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv4_2')(acti_nconv4_1)
        bn_nconv4_2 = BatchNormalization()(nconv4_2)
        acti_nconv4_2 = Activation(activation='relu')(bn_nconv4_2)

        # ###====== Deconv + Conv ======####
        ####################################

        acti_nconv2_up = UpSampling2D(size=(2, 2))(acti_nconv2_2)
        nconv2_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                           name='nconv2_up')(acti_nconv2_up)
        nconv2_up = BatchNormalization()(nconv2_up)
        nconv2_up = Activation(activation='relu')(nconv2_up)

        acti_nconv3_up = UpSampling2D(size=(4, 4))(acti_nconv3_2)
        nconv3_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                           name='nconv3_up')(acti_nconv3_up)
        nconv3_up = BatchNormalization()(nconv3_up)
        nconv3_up = Activation(activation='relu')(nconv3_up)

        acti_nconv4_up = UpSampling2D(size=(8, 8))(acti_nconv4_2)
        nconv4_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                           name='nconv4_up')(acti_nconv4_up)
        nconv4_up = BatchNormalization()(nconv4_up)
        nconv4_up = Activation(activation='relu')(nconv4_up)

        conv9_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                          name='conv9_up')(acti9_2)
        conv9_up = BatchNormalization()(conv9_up)
        conv9_up = Activation(activation='relu')(conv9_up)

        conv10 = Add()([nconv2_up, nconv3_up, nconv4_up, conv9_up])
        output = Conv2D(filters=self.classes, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                        name='output', activation='softmax')(conv10)

        # conv10 = Conv2D(filters=self.classes, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
        #                 name='conv10')(acti9_2)
        #

        model = Model(inputs=self.input, outputs=output)

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
        print("###数据增强前###", X_train.shape, Y_train.shape)

        if mode:
            X_tmp = rotate(X_train[0:6000], 90, (1, 2))  # 作旋转进行数据增强吗
            X_tmp = np.append(X_tmp, rotate(X_train[6000:], -90, (1, 2)), axis=0)
            X_train = np.append(X_train, X_tmp, axis=0)

            Y_tmp = rotate(Y_train[0:6000], 90, (1, 2))
            Y_tmp = np.append(Y_tmp, rotate(Y_train[6000:], -90, (1, 2)), axis=0)
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
        model = self.get_model()
        model.summary()
        # plot_model(model, to_file='./DUnet.png', show_shapes=True, show_layer_names=True)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)

        change_lr = LearningRateScheduler(self.step_decay)
        sgd = SGD(lr=0.003, momentum=0.9, decay=0, nesterov=True)
        adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', self.dice_comp,
                                                                                  self.dice_core, self.dice_en])
        checkpointer = ModelCheckpoint(filepath='../weights/PA+FP/3/deconv_150_5.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=70, batch_size=16,
                         shuffle=True, callbacks=[checkpointer, change_lr])
        with open('../weights/PA+FP/3/PA+FP_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)

        # model.fit(X_train, Y_train,
        #           validation_data=(X_test, Y_test),
        #           epochs=70,
        #           batch_size=8,
        #           shuffle=True,
        #           callbacks=[TensorBoard(log_dir='./log'), checkpointer])

    def test(self):
        model = self.conv_block()
        model.summary()
        model.load_weights('./PA+FP0/weights/dunet_150_5.69-0.03.hdf5')
        X_train, Y_train, X_test, Y_test = self.load_data('./data/BraTS15_all/', mode=False)
        del X_train, Y_train, Y_test
        model.summary()
        print(X_test.shape)
        pred = model.predict(X_test, verbose=1)
        np.save('./dunet_tfcv5_5.npy', pred)
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
        np.save('./dunet_tfcv5_1.npy', pred1)

class FPN4:

    def __init__(self, inputshape=(160, 160, 4), classes=4):
        self.input = Input(inputshape)
        self.classes = classes

    def get_model(self):

        # 160 160 4
        conv1_1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv1_1')(self.input)
        bn1_1 = BatchNormalization()(conv1_1)
        acti1_1 = Activation(activation='relu')(bn1_1)

        conv1_2 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv1_2')(acti1_1)
        bn1_2 = BatchNormalization()(conv1_2)
        acti1_2 = Activation(activation='relu')(bn1_2)
        maxpool1 = MaxPool2D()(acti1_2)

        conv2_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv2_1')(maxpool1)
        bn2_1 = BatchNormalization()(conv2_1)
        acti2_1 = Activation(activation='relu')(bn2_1)

        conv2_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv2_2')(acti2_1)
        bn2_2 = BatchNormalization()(conv2_2)
        acti2_2 = Activation(activation='relu')(bn2_2)
        maxpool2 = MaxPool2D()(acti2_2)

        conv3_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv3_1')(maxpool2)
        bn3_1 = BatchNormalization()(conv3_1)
        acti3_1 = Activation(activation='relu')(bn3_1)

        conv3_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv3_2')(acti3_1)
        bn3_2 = BatchNormalization()(conv3_2)
        acti3_2 = Activation(activation='relu')(bn3_2)
        maxpool3 = MaxPool2D()(acti3_2)

        conv4_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv4_1')(maxpool3)
        bn4_1 = BatchNormalization()(conv4_1)
        acti4_1 = Activation(activation='relu')(bn4_1)

        conv4_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv4_2')(acti4_1)
        bn4_2 = BatchNormalization()(conv4_2)
        acti4_2 = Activation(activation='relu')(bn4_2)
        maxpool4 = MaxPool2D()(acti4_2)

        conv5_1 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv5_1')(maxpool4)
        bn5_1 = BatchNormalization()(conv5_1)
        acti5_1 = Activation(activation='relu')(bn5_1)

        conv5_2 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv5_2')(acti5_1)
        bn5_2 = BatchNormalization()(conv5_2)
        acti5_2 = Activation(activation='relu')(bn5_2)

        up6 = Conv2D(filters=256, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up6')(UpSampling2D()(acti5_2))

        concat6 = Concatenate(axis=-1)([up6, acti4_2])

        conv6_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv6_1')(concat6)
        bn6_1 = BatchNormalization()(conv6_1)
        acti6_1 = Activation(activation='relu')(bn6_1)

        conv6_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv6_2')(acti6_1)
        bn6_2 = BatchNormalization()(conv6_2)
        acti6_2 = Activation(activation='relu')(bn6_2)

        up7 = Conv2D(filters=128, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up7')(UpSampling2D()(acti6_2))

        concat7 = Concatenate(axis=-1)([up7, acti3_2])

        conv7_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv7_1')(concat7)
        bn7_1 = BatchNormalization()(conv7_1)
        acti7_1 = Activation(activation='relu')(bn7_1)

        conv7_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv7_2')(acti7_1)
        bn7_2 = BatchNormalization()(conv7_2)
        acti7_2 = Activation(activation='relu')(bn7_2)

        up8 = Conv2D(filters=64, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up8')(UpSampling2D()(acti7_2))

        concat8 = Concatenate(axis=-1)([up8, acti2_2])

        conv8_1 = Conv2D(filters=64, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv8_1')(concat8)
        bn8_1 = BatchNormalization()(conv8_1)
        acti8_1 = Activation(activation='relu')(bn8_1)

        conv8_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv8_2')(acti8_1)
        bn8_2 = BatchNormalization()(conv8_2)
        acti8_2 = Activation(activation='relu')(bn8_2)

        up9 = Conv2D(filters=32, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up9')(UpSampling2D()(acti8_2))

        concat9 = Concatenate(axis=-1)([up9, acti1_2])

        conv9_1 = Conv2D(filters=32, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv9_1')(concat9)
        bn9_1 = BatchNormalization()(conv9_1)
        acti9_1 = Activation(activation='relu')(bn9_1)

        conv9_2 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv9_2')(acti9_1)
        bn9_2 = BatchNormalization()(conv9_2)
        acti9_2 = Activation(activation='relu')(bn9_2)

        # ==================================nconv2=====================================#
        down2 = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal',
                       name='down2')(acti9_2)
        bn_down2 = BatchNormalization()(down2)
        acti_down2 = Activation(activation='relu')(bn_down2)

        concat2 = Concatenate(axis=-1)([acti_down2, acti8_2])

        nconv2_1 = Conv2D(filters=64, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv2_1')(concat2)
        bn_nconv2_1 = BatchNormalization()(nconv2_1)
        acti_nconv2_1 = Activation(activation='relu')(bn_nconv2_1)

        nconv2_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv2_2')(acti_nconv2_1)
        bn_nconv2_2 = BatchNormalization()(nconv2_2)
        acti_nconv2_2 = Activation(activation='relu')(bn_nconv2_2)

        # ###====== Deconv + Conv ======####

        # =================================nconv3=======================================#

        down3 = Conv2D(filters=128, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal',
                       name='down3')(acti_nconv2_2)
        bn_down3 = BatchNormalization()(down3)
        acti_down3 = Activation(activation='relu')(bn_down3)

        concat3 = Concatenate(axis=-1)([acti_down3, acti7_2])

        nconv3_1 = Conv2D(filters=128, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv3_1')(concat3)
        bn_nconv3_1 = BatchNormalization()(nconv3_1)
        acti_nconv3_1 = Activation(activation='relu')(bn_nconv3_1)

        nconv3_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv3_2')(acti_nconv3_1)
        bn_nconv3_2 = BatchNormalization()(nconv3_2)
        acti_nconv3_2 = Activation(activation='relu')(bn_nconv3_2)

        # ###====== Deconv + Conv ======####
        ####################################

        # =================================nconv4=======================================#

        down4 = Conv2D(filters=256, kernel_size=3, strides=2, padding='same', kernel_initializer='he_normal',
                       name='down4')(acti_nconv3_2)
        bn_down4 = BatchNormalization()(down4)
        acti_down4 = Activation(activation='relu')(bn_down4)

        concat4 = Concatenate(axis=-1)([acti_down4, acti6_2])

        nconv4_1 = Conv2D(filters=256, kernel_size=2, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv4_1')(concat4)
        bn_nconv4_1 = BatchNormalization()(nconv4_1)
        acti_nconv4_1 = Activation(activation='relu')(bn_nconv4_1)

        nconv4_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                          name='nconv4_2')(acti_nconv4_1)
        bn_nconv4_2 = BatchNormalization()(nconv4_2)
        acti_nconv4_2 = Activation(activation='relu')(bn_nconv4_2)

        # ###====== Deconv + Conv ======####
        ####################################

        acti_nconv2_up = UpSampling2D(size=(2, 2))(acti_nconv2_2)
        nconv2_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                           name='nconv2_up')(acti_nconv2_up)
        nconv2_up = BatchNormalization()(nconv2_up)
        nconv2_up = Activation(activation='relu')(nconv2_up)

        acti_nconv3_up = UpSampling2D(size=(4, 4))(acti_nconv3_2)
        nconv3_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                           name='nconv3_up')(acti_nconv3_up)
        nconv3_up = BatchNormalization()(nconv3_up)
        nconv3_up = Activation(activation='relu')(nconv3_up)

        acti_nconv4_up = UpSampling2D(size=(8, 8))(acti_nconv4_2)
        nconv4_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                           name='nconv4_up')(acti_nconv4_up)
        nconv4_up = BatchNormalization()(nconv4_up)
        nconv4_up = Activation(activation='relu')(nconv4_up)

        conv9_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                          name='conv9_up')(acti9_2)
        conv9_up = BatchNormalization()(conv9_up)
        conv9_up = Activation(activation='relu')(conv9_up)

        conv10 = Add()([nconv2_up, nconv3_up, nconv4_up, conv9_up])
        output = Conv2D(filters=self.classes, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                        name='output', activation='softmax')(conv10)

        # conv10 = Conv2D(filters=self.classes, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
        #                 name='conv10')(acti9_2)
        #

        model = Model(inputs=self.input, outputs=output)

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
        print("###数据增强前###", X_train.shape, Y_train.shape)

        if mode:
            X_tmp = rotate(X_train[0:6000], 90, (1, 2))  # 作旋转进行数据增强吗
            X_tmp = np.append(X_tmp, rotate(X_train[6000:], -90, (1, 2)), axis=0)
            X_train = np.append(X_train, X_tmp, axis=0)

            Y_tmp = rotate(Y_train[0:6000], 90, (1, 2))
            Y_tmp = np.append(Y_tmp, rotate(Y_train[6000:], -90, (1, 2)), axis=0)
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
        model = self.get_model()
        model.summary()
        # plot_model(model, to_file='./DUnet.png', show_shapes=True, show_layer_names=True)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)

        change_lr = LearningRateScheduler(self.step_decay)
        sgd = SGD(lr=0.003, momentum=0.9, decay=0, nesterov=True)
        adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', self.dice_comp,
                                                                                  self.dice_core, self.dice_en])
        checkpointer = ModelCheckpoint(filepath='../weights/PA+FP/4/deconv_150_5.{epoch:02d}-{val_loss:.2f}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=70, batch_size=16,
                         shuffle=True, callbacks=[checkpointer, change_lr])
        with open('../weights/PA+FP/4/PA+FP_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)

        # model.fit(X_train, Y_train,
        #           validation_data=(X_test, Y_test),
        #           epochs=70,
        #           batch_size=8,
        #           shuffle=True,
        #           callbacks=[TensorBoard(log_dir='./log'), checkpointer])

    def test(self):
        model = self.conv_block()
        model.summary()
        model.load_weights('./PA+FP4/weights/dunet_150_5.69-0.03.hdf5')
        X_train, Y_train, X_test, Y_test = self.load_data('./data/BraTS15_all/', mode=False)
        del X_train, Y_train, Y_test
        model.summary()
        print(X_test.shape)
        pred = model.predict(X_test, verbose=1)
        np.save('./dunet_tfcv5_5.npy', pred)
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
        np.save('./dunet_tfcv5_1.npy', pred1)







if __name__ == '__main__':
    # dunet = DUnet()
    #
    # dunet.train()
    # #dunet.test()
    fpn0 = FPN0()
    fpn0.train()

    # fpn1 = FPN1()
    # fpn1.train()
    #
    # fpn2 = FPN2()
    # fpn2.train()
    #
    # fpn3 = FPN3()
    # fpn3.train()
    #
    # fpn4 = FPN4()
    # fpn4.train()