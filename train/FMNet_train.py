import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from keras.layers import *
# Conv2D, BatchNormalization, MaxPool2D, Concatenate, Input, Activation, UpSampling2D, Add, Multiply
from keras.models import Model
import numpy as np
from keras.utils import plot_model
from sklearn.utils import shuffle
from scipy.ndimage.interpolation import rotate
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler, TensorBoard
from keras.optimizers import Adam, SGD
import json, os

np.random.seed(1234)

def ConvBNActi(x, filters, kernel_size, strides=1, padding='same', kernel_initializer='he_normal'):
    conv = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                  kernel_initializer=kernel_initializer)(x)
    bn = BatchNormalization()(conv)
    acti = Activation(activation='relu')(bn)
    return acti


def feature_correction_1(acti_n, up_n):
    concat = Concatenate()([acti_n, up_n])
    return concat





# brats2017
class FPN:

    def __init__(self, inputshape=(160, 160, 4), classes=4):
        self.input = Input(inputshape)
        self.classes = classes

    def get_model(self):
        # layer 1
        conv1_1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv1_1')(self.input)
        bn1_1 = BatchNormalization()(conv1_1)
        acti1_1 = Activation(activation='relu')(bn1_1)

        conv1_2 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv1_2')(acti1_1)
        bn1_2 = BatchNormalization()(conv1_2)
        acti1_2 = Activation(activation='relu')(bn1_2)

        conv1_3 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv1_3')(self.input)
        bn1_3 = BatchNormalization()(conv1_3)
        acti1_3 = Activation(activation='relu')(bn1_3)
        acti1_2 = Add()([acti1_3, acti1_2])
        maxpool1 = MaxPool2D()(acti1_2)
        # layer 2
        conv2_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv2_1')(maxpool1)
        bn2_1 = BatchNormalization()(conv2_1)
        acti2_1 = Activation(activation='relu')(bn2_1)

        conv2_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv2_2')(acti2_1)
        bn2_2 = BatchNormalization()(conv2_2)
        acti2_2 = Activation(activation='relu')(bn2_2)

        conv2_3 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv2_3')(maxpool1)
        bn2_3 = BatchNormalization()(conv2_3)
        acti2_3 = Activation(activation='relu')(bn2_3)
        acti2_2 = Add()([acti2_2, acti2_3])
        maxpool2 = MaxPool2D()(acti2_2)
        # layer 3
        conv3_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv3_1')(maxpool2)
        bn3_1 = BatchNormalization()(conv3_1)
        acti3_1 = Activation(activation='relu')(bn3_1)

        conv3_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv3_2')(acti3_1)
        bn3_2 = BatchNormalization()(conv3_2)
        acti3_2 = Activation(activation='relu')(bn3_2)

        conv3_3 = Conv2D(filters=128, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv3_3')(maxpool2)
        bn3_3 = BatchNormalization()(conv3_3)
        acti3_3 = Activation(activation='relu')(bn3_3)
        acti3_2 = Add()([acti3_2, acti3_3])
        maxpool3 = MaxPool2D()(acti3_2)
        # layer 4
        conv4_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv4_1')(maxpool3)
        bn4_1 = BatchNormalization()(conv4_1)
        acti4_1 = Activation(activation='relu')(bn4_1)

        conv4_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv4_2')(acti4_1)
        bn4_2 = BatchNormalization()(conv4_2)
        acti4_2 = Activation(activation='relu')(bn4_2)

        conv4_3 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv4_3')(maxpool3)
        bn4_3 = BatchNormalization()(conv4_3)
        acti4_3 = Activation(activation='relu')(bn4_3)
        acti4_2 = Add()([acti4_2, acti4_3])
        maxpool4 = MaxPool2D()(acti4_2)
        # layer 5
        conv5_1 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv5_1')(maxpool4)
        bn5_1 = BatchNormalization()(conv5_1)
        acti5_1 = Activation(activation='relu')(bn5_1)

        conv5_2 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv5_2')(acti5_1)
        bn5_2 = BatchNormalization()(conv5_2)
        acti5_2 = Activation(activation='relu')(bn5_2)

        conv5_3 = Conv2D(filters=512, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv5_3')(maxpool4)
        bn5_3 = BatchNormalization()(conv5_3)
        acti5_3 = Activation(activation='relu')(bn5_3)
        acti5_2 = Add()([acti5_2, acti5_3])
        # layer 4_2

        up6 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up6')(UpSampling2D()(acti5_2))

        concat6 = feature_correction_1(acti4_2, up6)
        conv6_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv6_1')(concat6)
        bn6_1 = BatchNormalization()(conv6_1)
        acti6_1 = Activation(activation='relu')(bn6_1)
        conv6_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv6_2')(acti6_1)
        bn6_2 = BatchNormalization()(conv6_2)
        acti6_2 = Activation(activation='relu')(bn6_2)

        conv6_3 = Conv2D(filters=256, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv6_3')(concat6)
        bn6_3 = BatchNormalization()(conv6_3)
        acti6_3 = Activation(activation='relu')(bn6_3)
        acti6_2 = Add()([acti6_2, acti6_3])

        # layer 3_2
        up7 = Conv2D(filters=128, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up7')(UpSampling2D()(acti6_2))

        concat7 = feature_correction_1(acti3_2, up7)
        conv7_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv7_1')(concat7)
        bn7_1 = BatchNormalization()(conv7_1)
        acti7_1 = Activation(activation='relu')(bn7_1)
        conv7_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv7_2')(acti7_1)
        bn7_2 = BatchNormalization()(conv7_2)
        acti7_2 = Activation(activation='relu')(bn7_2)

        conv7_3 = Conv2D(filters=128, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv7_3')(concat7)
        bn7_3 = BatchNormalization()(conv7_3)
        acti7_3 = Activation(activation='relu')(bn7_3)
        acti7_2 = Add()([acti7_2, acti7_3])

        # layer2_2
        up8 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up8')(UpSampling2D()(acti7_2))
        concat8 = feature_correction_1(acti2_2, up8)
        conv8_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv8_1')(concat8)
        bn8_1 = BatchNormalization()(conv8_1)
        acti8_1 = Activation(activation='relu')(bn8_1)
        conv8_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv8_2')(acti8_1)
        bn8_2 = BatchNormalization()(conv8_2)
        acti8_2 = Activation(activation='relu')(bn8_2)

        conv8_3 = Conv2D(filters=64, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv8_3')(concat8)
        bn8_3 = BatchNormalization()(conv8_3)
        acti8_3 = Activation(activation='relu')(bn8_3)
        acti8_2 = Add()([acti8_2, acti8_3])

        # layer 1_2
        up9 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                     name='up9')(UpSampling2D()(acti8_2))
        concat9 = feature_correction_1(acti1_2, up9)
        conv9_1 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv9_1')(concat9)
        bn9_1 = BatchNormalization()(conv9_1)
        acti9_1 = Activation(activation='relu')(bn9_1)
        conv9_2 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv9_2')(acti9_1)
        bn9_2 = BatchNormalization()(conv9_2)
        acti9_2 = Activation(activation='relu')(bn9_2)

        conv9_3 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                         name='conv9_3')(concat9)
        bn9_3 = BatchNormalization()(conv9_3)
        acti9_3 = Activation(activation='relu')(bn9_3)
        acti9_2 = Add()([acti9_2, acti9_3])
        # PA layer
        acti6_up = UpSampling2D(size=(8, 8))(acti6_2)
        conv6_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                          name='conv6_up')(acti6_up)
        conv6_up = BatchNormalization()(conv6_up)
        conv6_up = Activation(activation='relu')(conv6_up)

        acti7_up = UpSampling2D(size=(4, 4))(acti7_2)
        conv7_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                          name='conv7_up')(acti7_up)
        conv7_up = BatchNormalization()(conv7_up)
        conv7_up = Activation(activation='relu')(conv7_up)

        acti8_up = UpSampling2D(size=(2, 2))(acti8_2)
        conv8_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                          name='conv8_up')(acti8_up)
        conv8_up = BatchNormalization()(conv8_up)
        conv8_up = Activation(activation='relu')(conv8_up)

        conv9_up = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                          name='conv9_up')(acti9_2)
        conv9_up = BatchNormalization()(conv9_up)
        conv9_up = Activation(activation='relu')(conv9_up)

        conv10 = Add()([conv6_up, conv7_up, conv8_up, conv9_up])
        output = Conv2D(filters=self.classes, kernel_size=1, strides=1, padding='same', kernel_initializer='he_normal',
                        name='output', activation='softmax')(conv10)

        model = Model(inputs=self.input, outputs=output)

        return model



    def step_decay(slef, epochs):
        #init_rate = 0.01
        #fin_rate = 0.001
        #total_epochs = 20
        if epochs < 30:
            lrate = 0.001
        elif epochs >= 30 and epochs < 50:
            lrate = 0.0001
            #lrate = init_rate - (init_rate - fin_rate) / total_epochs * float(epochs)
        else:
            lrate = 0.00003
        return lrate

        #init_rate = 0.001
        #fin_rate = 0.00003
        #total_epochs = 24
        #if epochs < 25:
        #    lrate = init_rate - (init_rate - fin_rate) / total_epochs * float(epochs)
        #else:
        #    lrate = 0.00003
        #return lrate

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
        #model.summary()

        tbCallBack = TensorBoard(log_dir='../Graph/aunet47',
                                         histogram_freq=0,
                                         write_graph=True, 
                                         write_images=True)#画图

        X_train, Y_train, X_test, Y_test = self.load_data('../data/')
        change_lr = LearningRateScheduler(self.step_decay)
        sgd = SGD(lr=0.003, momentum=0.9, decay=0, nesterov=True)
        adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy', self.dice_comp,
                                                                                  self.dice_core, self.dice_en])
        checkpointer = ModelCheckpoint(filepath='../weights/aunet47/fpn.{epoch:02d}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=70, batch_size=6,
                         shuffle=True, callbacks=[checkpointer, change_lr, tbCallBack], verbose=1)
        with open('../weights/aunet47/fpn.json', 'w') as f:
            json.dump(hist.history, f)

    def test(self):
        model = self.get_model()
        model.summary()
        model.load_weights('../weights/brats2018/aunet47/fpn.70.hdf5')
        # X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        X_train, Y_train, X_test, Y_test = self.load_data('../../PAUnet-master-2018/data/', mode=False)
        del X_train, Y_train, Y_test
        model.summary()
        print(X_test.shape)
        pred = model.predict(X_test, verbose=1)
        # np.save('./dunet_tfcv5_5.npy', pred)
        print(pred.shape)
        print('#### start test ###')
        pred1 = np.argmax(pred, axis=3)
        print(pred1.shape)
        np.save('../results/brats2018/aunet47/fpn.npy', pred1)


if __name__ == '__main__':
    fpn = FPN(inputshape=(240, 240, 4), classes=4)
    # fpn.train()
    fpn.test()
    # model = fpn.get_model()
    # print(model.summary())
    # plot_model(model, to_file='./FPN.png', show_layer_names=True, show_shapes=True)


