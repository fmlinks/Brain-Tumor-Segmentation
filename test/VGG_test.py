import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.layers import Input, Conv2D, MaxPool2D, UpSampling2D, Concatenate, BatchNormalization, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from scipy.ndimage.interpolation import rotate
from keras import backend as K
import json
import tensorflow as tf
from keras.optimizers import Adam, SGD
import numpy as np
from keras.utils import plot_model
from sklearn.utils import shuffle

np.random.seed(1234)



class VGG0():
    def conv_block(self, x, block, nb_filters, kernel):
        base_name = 'block' + str(block) + '_'
        x = Conv2D(nb_filters, kernel, padding='same', activation='relu', name=base_name + 'conv1')(x)
        x = Conv2D(nb_filters, kernel, padding='same', activation='relu', name=base_name + 'conv2')(x)
        if (block > 2):
            x = Conv2D(nb_filters + 64, kernel, padding='same', activation='relu', name=base_name + 'conv3')(x)
            x = MaxPool2D(pool_size=(2, 2), name=base_name + 'pool')(x)
        else:
            x = MaxPool2D(pool_size=(2, 2), name=base_name + 'pool')(x)
        return x

    def get_model(self, block=5, nb_filters=64, kernel=3):
        input_tensor = Input((240, 240, 4))
        x = input_tensor
        # for each_block in range(1, block):
        #
        #     x = conv_block(x, each_block, nb_filters=nb_filters * each_block, kernel=kernel)
        #
        # model = Model(inputs=input_tensor, outputs=x)
        # model.summary()

        block1_conv1 = Conv2D(nb_filters, kernel, padding='same', kernel_initializer='he_normal',
                              name='block1_conv1')(x)
        block1_bn1 = BatchNormalization(name='block1_bn1')(block1_conv1)
        block1_acti1 = Activation(activation='relu', name='block1_acti1')(block1_bn1)

        block1_conv2 = Conv2D(nb_filters, kernel, padding='same', kernel_initializer='he_normal',
                              name='block1_conv2')(block1_acti1)
        block1_bn2 = BatchNormalization(name='block1_bn2')(block1_conv2)
        block1_acti2 = Activation(activation='relu', name='block1_acti2')(block1_bn2)

        block1_pool = MaxPool2D(pool_size=(2, 2), name='block1_pool')(block1_acti2)

        block2_conv1 = Conv2D(nb_filters * 2, kernel, padding='same', kernel_initializer='he_normal',
                              name='block2_conv1')(block1_pool)
        block2_bn1 = BatchNormalization(name='block2_bn1')(block2_conv1)
        block2_acti1 = Activation(activation='relu', name='block2_acti1')(block2_bn1)

        block2_conv2 = Conv2D(nb_filters * 2, kernel, padding='same', kernel_initializer='he_normal',
                              name='block2_conv2')(block2_acti1)
        block2_bn2 = BatchNormalization(name='block2_bn2')(block2_conv2)
        block2_acti2 = Activation(activation='relu', name='block2_acti2')(block2_bn2)

        block2_pool = MaxPool2D(pool_size=(2, 2), name='block2_pool')(block2_acti2)

        block3_conv1 = Conv2D(nb_filters * 4, kernel, padding='same', kernel_initializer='he_normal',
                              name='block3_conv1')(block2_pool)
        block3_bn1 = BatchNormalization(name='block3_bn1')(block3_conv1)
        block3_acti1 = Activation(activation='relu', name='block3_acti1')(block3_bn1)

        block3_conv2 = Conv2D(nb_filters * 4, kernel, padding='same', kernel_initializer='he_normal',
                              name='block3_conv2')(block3_acti1)
        block3_bn2 = BatchNormalization(name='block3_bn2')(block3_conv2)
        block3_acti2 = Activation(activation='relu', name='block3_acti2')(block3_bn2)

        block3_conv3 = Conv2D(nb_filters * 4, kernel, padding='same', kernel_initializer='he_normal',
                              name='block3_conv3')(block3_acti2)
        block3_bn3 = BatchNormalization(name='block3_bn3')(block3_conv3)
        block3_acti3 = Activation(activation='relu', name='block3_acti3')(block3_bn3)

        up_conv1 = Conv2D(nb_filters, kernel, activation='relu', padding='same', kernel_initializer='he_normal',
                          name='up_conv1')(block1_acti2)

        up1 = UpSampling2D(size=(2, 2), name='up1')(block2_acti2)
        up_conv2 = Conv2D(nb_filters, kernel, activation='relu', padding='same', kernel_initializer='he_normal',
                          name='up_conv2')(up1)

        up2 = UpSampling2D(size=(4, 4), name='up2')(block3_acti3)
        up_conv3 = Conv2D(nb_filters, kernel, activation='relu', padding='same', kernel_initializer='he_normal',
                          name='up_conv3')(up2)

        concat = Concatenate(axis=3)([up_conv1, up_conv2, up_conv3])
        output = Conv2D(4, 1, activation='softmax', kernel_initializer='he_normal',
                        name='output')(concat)

        model = Model(inputs=x, outputs=output)
        model.summary()
        return model

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

    def step_decay(self, epochs):
        # init_rate = 0.01
        # fin_rate = 0.001
        # total_epochs = 20
        if epochs < 30:
            lrate = 0.001
        elif epochs >= 30 and epochs < 50:
            lrate = 0.0001
            # lrate = init_rate - (init_rate - fin_rate) / total_epochs * float(epochs)
        else:
            lrate = 0.00003
        return lrate

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

        print(model.summary())
        change_lr = LearningRateScheduler(self.step_decay)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)

        sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        # model.load_weights('../weights/FCDense_tfcv4.22-0.0269.hdf5')
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer=adam, loss='categorical_crossentropy',
                      metrics=['accuracy', self.dice_comp, self.dice_core, self.dice_en])  # adam 0.001 SGD 0.0001 SGD 0.00001
        checkpointer = ModelCheckpoint(filepath='../weights/VGG/0/VGG16_tfcv5.{epoch:02d}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=70, batch_size=16, shuffle=True,
                         callbacks=[checkpointer, change_lr])

        with open('../weights/VGG/0/VGG16_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)


    def test(self):
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        print(X_test.shape)
        model = self.get_model()
        model.load_weights('../weights/VGG/0/VGG16_tfcv5.70.hdf5')
        pred = model.predict(X_test, batch_size=4, verbose=1)
        # np.save('./VGG0/VGG16_tfcv5_5.npy', pred)
        print(pred.shape)
        print('#### start test ###')
        pred1 = np.argmax(pred, axis=3)
        print(pred1.shape)
        np.save('../results/VGG/0/vgg_test.npy', pred1)


class VGG1():
    def conv_block(self, x, block, nb_filters, kernel):
        base_name = 'block' + str(block) + '_'
        x = Conv2D(nb_filters, kernel, padding='same', activation='relu', name=base_name + 'conv1')(x)
        x = Conv2D(nb_filters, kernel, padding='same', activation='relu', name=base_name + 'conv2')(x)
        if (block > 2):
            x = Conv2D(nb_filters + 64, kernel, padding='same', activation='relu', name=base_name + 'conv3')(x)
            x = MaxPool2D(pool_size=(2, 2), name=base_name + 'pool')(x)
        else:
            x = MaxPool2D(pool_size=(2, 2), name=base_name + 'pool')(x)
        return x

    def get_model(self, block=5, nb_filters=64, kernel=3):
        input_tensor = Input((240, 240, 4))
        x = input_tensor
        # for each_block in range(1, block):
        #
        #     x = conv_block(x, each_block, nb_filters=nb_filters * each_block, kernel=kernel)
        #
        # model = Model(inputs=input_tensor, outputs=x)
        # model.summary()

        block1_conv1 = Conv2D(nb_filters, kernel, padding='same', kernel_initializer='he_normal',
                              name='block1_conv1')(x)
        block1_bn1 = BatchNormalization(name='block1_bn1')(block1_conv1)
        block1_acti1 = Activation(activation='relu', name='block1_acti1')(block1_bn1)

        block1_conv2 = Conv2D(nb_filters, kernel, padding='same', kernel_initializer='he_normal',
                              name='block1_conv2')(block1_acti1)
        block1_bn2 = BatchNormalization(name='block1_bn2')(block1_conv2)
        block1_acti2 = Activation(activation='relu', name='block1_acti2')(block1_bn2)

        block1_pool = MaxPool2D(pool_size=(2, 2), name='block1_pool')(block1_acti2)

        block2_conv1 = Conv2D(nb_filters * 2, kernel, padding='same', kernel_initializer='he_normal',
                              name='block2_conv1')(block1_pool)
        block2_bn1 = BatchNormalization(name='block2_bn1')(block2_conv1)
        block2_acti1 = Activation(activation='relu', name='block2_acti1')(block2_bn1)

        block2_conv2 = Conv2D(nb_filters * 2, kernel, padding='same', kernel_initializer='he_normal',
                              name='block2_conv2')(block2_acti1)
        block2_bn2 = BatchNormalization(name='block2_bn2')(block2_conv2)
        block2_acti2 = Activation(activation='relu', name='block2_acti2')(block2_bn2)

        block2_pool = MaxPool2D(pool_size=(2, 2), name='block2_pool')(block2_acti2)

        block3_conv1 = Conv2D(nb_filters * 4, kernel, padding='same', kernel_initializer='he_normal',
                              name='block3_conv1')(block2_pool)
        block3_bn1 = BatchNormalization(name='block3_bn1')(block3_conv1)
        block3_acti1 = Activation(activation='relu', name='block3_acti1')(block3_bn1)

        block3_conv2 = Conv2D(nb_filters * 4, kernel, padding='same', kernel_initializer='he_normal',
                              name='block3_conv2')(block3_acti1)
        block3_bn2 = BatchNormalization(name='block3_bn2')(block3_conv2)
        block3_acti2 = Activation(activation='relu', name='block3_acti2')(block3_bn2)

        block3_conv3 = Conv2D(nb_filters * 4, kernel, padding='same', kernel_initializer='he_normal',
                              name='block3_conv3')(block3_acti2)
        block3_bn3 = BatchNormalization(name='block3_bn3')(block3_conv3)
        block3_acti3 = Activation(activation='relu', name='block3_acti3')(block3_bn3)

        up_conv1 = Conv2D(nb_filters, kernel, activation='relu', padding='same', kernel_initializer='he_normal',
                          name='up_conv1')(block1_acti2)

        up1 = UpSampling2D(size=(2, 2), name='up1')(block2_acti2)
        up_conv2 = Conv2D(nb_filters, kernel, activation='relu', padding='same', kernel_initializer='he_normal',
                          name='up_conv2')(up1)

        up2 = UpSampling2D(size=(4, 4), name='up2')(block3_acti3)
        up_conv3 = Conv2D(nb_filters, kernel, activation='relu', padding='same', kernel_initializer='he_normal',
                          name='up_conv3')(up2)

        concat = Concatenate(axis=3)([up_conv1, up_conv2, up_conv3])
        output = Conv2D(4, 1, activation='softmax', kernel_initializer='he_normal',
                        name='output')(concat)

        model = Model(inputs=x, outputs=output)
        model.summary()
        return model

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

    def step_decay(self, epochs):
        # init_rate = 0.01
        # fin_rate = 0.001
        # total_epochs = 20
        if epochs < 30:
            lrate = 0.001
        elif epochs >= 30 and epochs < 50:
            lrate = 0.0001
            # lrate = init_rate - (init_rate - fin_rate) / total_epochs * float(epochs)
        else:
            lrate = 0.00003
        return lrate

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

        print(model.summary())
        change_lr = LearningRateScheduler(self.step_decay)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)

        sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        # model.load_weights('../weights/FCDense_tfcv4.22-0.0269.hdf5')
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer=adam, loss='categorical_crossentropy',
                      metrics=['accuracy', self.dice_comp, self.dice_core, self.dice_en])  # adam 0.001 SGD 0.0001 SGD 0.00001
        checkpointer = ModelCheckpoint(filepath='../weights/VGG/1/VGG16_tfcv5.{epoch:02d}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, epochs=70, validation_data=(X_test, Y_test), batch_size=16, shuffle=True,
                         callbacks=[checkpointer, change_lr])

        with open('../weights/VGG/1/VGG16_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)

    def test(self):
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        print(X_test.shape)
        model = self.get_model()
        model.load_weights('../weights/VGG/1/VGG16_tfcv5.70.hdf5')
        pred = model.predict(X_test, batch_size=4, verbose=1)
        # np.save('./VGG1/VGG16_tfcv5_5.npy', pred)
        print(pred.shape)
        print('#### start test ###')
        pred1 = np.argmax(pred, axis=3)
        print(pred1.shape)
        np.save('../results/VGG/1/vgg_test.npy', pred1)


class VGG2():
    def conv_block(self, x, block, nb_filters, kernel):
        base_name = 'block' + str(block) + '_'
        x = Conv2D(nb_filters, kernel, padding='same', activation='relu', name=base_name + 'conv1')(x)
        x = Conv2D(nb_filters, kernel, padding='same', activation='relu', name=base_name + 'conv2')(x)
        if (block > 2):
            x = Conv2D(nb_filters + 64, kernel, padding='same', activation='relu', name=base_name + 'conv3')(x)
            x = MaxPool2D(pool_size=(2, 2), name=base_name + 'pool')(x)
        else:
            x = MaxPool2D(pool_size=(2, 2), name=base_name + 'pool')(x)
        return x

    def get_model(self, block=5, nb_filters=64, kernel=3):
        input_tensor = Input((240, 240, 4))
        x = input_tensor
        # for each_block in range(1, block):
        #
        #     x = conv_block(x, each_block, nb_filters=nb_filters * each_block, kernel=kernel)
        #
        # model = Model(inputs=input_tensor, outputs=x)
        # model.summary()

        block1_conv1 = Conv2D(nb_filters, kernel, padding='same', kernel_initializer='he_normal',
                              name='block1_conv1')(x)
        block1_bn1 = BatchNormalization(name='block1_bn1')(block1_conv1)
        block1_acti1 = Activation(activation='relu', name='block1_acti1')(block1_bn1)

        block1_conv2 = Conv2D(nb_filters, kernel, padding='same', kernel_initializer='he_normal',
                              name='block1_conv2')(block1_acti1)
        block1_bn2 = BatchNormalization(name='block1_bn2')(block1_conv2)
        block1_acti2 = Activation(activation='relu', name='block1_acti2')(block1_bn2)

        block1_pool = MaxPool2D(pool_size=(2, 2), name='block1_pool')(block1_acti2)

        block2_conv1 = Conv2D(nb_filters * 2, kernel, padding='same', kernel_initializer='he_normal',
                              name='block2_conv1')(block1_pool)
        block2_bn1 = BatchNormalization(name='block2_bn1')(block2_conv1)
        block2_acti1 = Activation(activation='relu', name='block2_acti1')(block2_bn1)

        block2_conv2 = Conv2D(nb_filters * 2, kernel, padding='same', kernel_initializer='he_normal',
                              name='block2_conv2')(block2_acti1)
        block2_bn2 = BatchNormalization(name='block2_bn2')(block2_conv2)
        block2_acti2 = Activation(activation='relu', name='block2_acti2')(block2_bn2)

        block2_pool = MaxPool2D(pool_size=(2, 2), name='block2_pool')(block2_acti2)

        block3_conv1 = Conv2D(nb_filters * 4, kernel, padding='same', kernel_initializer='he_normal',
                              name='block3_conv1')(block2_pool)
        block3_bn1 = BatchNormalization(name='block3_bn1')(block3_conv1)
        block3_acti1 = Activation(activation='relu', name='block3_acti1')(block3_bn1)

        block3_conv2 = Conv2D(nb_filters * 4, kernel, padding='same', kernel_initializer='he_normal',
                              name='block3_conv2')(block3_acti1)
        block3_bn2 = BatchNormalization(name='block3_bn2')(block3_conv2)
        block3_acti2 = Activation(activation='relu', name='block3_acti2')(block3_bn2)

        block3_conv3 = Conv2D(nb_filters * 4, kernel, padding='same', kernel_initializer='he_normal',
                              name='block3_conv3')(block3_acti2)
        block3_bn3 = BatchNormalization(name='block3_bn3')(block3_conv3)
        block3_acti3 = Activation(activation='relu', name='block3_acti3')(block3_bn3)

        up_conv1 = Conv2D(nb_filters, kernel, activation='relu', padding='same', kernel_initializer='he_normal',
                          name='up_conv1')(block1_acti2)

        up1 = UpSampling2D(size=(2, 2), name='up1')(block2_acti2)
        up_conv2 = Conv2D(nb_filters, kernel, activation='relu', padding='same', kernel_initializer='he_normal',
                          name='up_conv2')(up1)

        up2 = UpSampling2D(size=(4, 4), name='up2')(block3_acti3)
        up_conv3 = Conv2D(nb_filters, kernel, activation='relu', padding='same', kernel_initializer='he_normal',
                          name='up_conv3')(up2)

        concat = Concatenate(axis=3)([up_conv1, up_conv2, up_conv3])
        output = Conv2D(4, 1, activation='softmax', kernel_initializer='he_normal',
                        name='output')(concat)

        model = Model(inputs=x, outputs=output)
        model.summary()
        return model

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

    def step_decay(self, epochs):
        # init_rate = 0.01
        # fin_rate = 0.001
        # total_epochs = 20
        if epochs < 30:
            lrate = 0.001
        elif epochs >= 30 and epochs < 50:
            lrate = 0.0001
            # lrate = init_rate - (init_rate - fin_rate) / total_epochs * float(epochs)
        else:
            lrate = 0.00003
        return lrate

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

        print(model.summary())
        change_lr = LearningRateScheduler(self.step_decay)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)

        sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        # model.load_weights('../weights/FCDense_tfcv4.22-0.0269.hdf5')
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer=adam, loss='categorical_crossentropy',
                      metrics=['accuracy', self.dice_comp, self.dice_core, self.dice_en])  # adam 0.001 SGD 0.0001 SGD 0.00001
        checkpointer = ModelCheckpoint(filepath='../weights/VGG/2/VGG16_tfcv5.{epoch:02d}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, epochs=70, validation_data=(X_test, Y_test), batch_size=16, shuffle=True,
                         callbacks=[checkpointer, change_lr])

        with open('../weights/VGG/2/VGG16_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)

    def test(self):
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        print(X_test.shape)
        model = self.get_model()
        model.load_weights('../weights/VGG/2/VGG16_tfcv5.70.hdf5')
        pred = model.predict(X_test, batch_size=4, verbose=1)
        # np.save('./VGG2/VGG16_tfcv5_5.npy', pred)
        print(pred.shape)
        print('#### start test ###')
        pred1 = np.argmax(pred, axis=3)
        print(pred1.shape)
        np.save('../results/VGG/2/vgg_test.npy', pred1)


class VGG3():
    def conv_block(self, x, block, nb_filters, kernel):
        base_name = 'block' + str(block) + '_'
        x = Conv2D(nb_filters, kernel, padding='same', activation='relu', name=base_name + 'conv1')(x)
        x = Conv2D(nb_filters, kernel, padding='same', activation='relu', name=base_name + 'conv2')(x)
        if (block > 2):
            x = Conv2D(nb_filters + 64, kernel, padding='same', activation='relu', name=base_name + 'conv3')(x)
            x = MaxPool2D(pool_size=(2, 2), name=base_name + 'pool')(x)
        else:
            x = MaxPool2D(pool_size=(2, 2), name=base_name + 'pool')(x)
        return x

    def get_model(self, block=5, nb_filters=64, kernel=3):
        input_tensor = Input((240, 240, 4))
        x = input_tensor
        # for each_block in range(1, block):
        #
        #     x = conv_block(x, each_block, nb_filters=nb_filters * each_block, kernel=kernel)
        #
        # model = Model(inputs=input_tensor, outputs=x)
        # model.summary()

        block1_conv1 = Conv2D(nb_filters, kernel, padding='same', kernel_initializer='he_normal',
                              name='block1_conv1')(x)
        block1_bn1 = BatchNormalization(name='block1_bn1')(block1_conv1)
        block1_acti1 = Activation(activation='relu', name='block1_acti1')(block1_bn1)

        block1_conv2 = Conv2D(nb_filters, kernel, padding='same', kernel_initializer='he_normal',
                              name='block1_conv2')(block1_acti1)
        block1_bn2 = BatchNormalization(name='block1_bn2')(block1_conv2)
        block1_acti2 = Activation(activation='relu', name='block1_acti2')(block1_bn2)

        block1_pool = MaxPool2D(pool_size=(2, 2), name='block1_pool')(block1_acti2)

        block2_conv1 = Conv2D(nb_filters * 2, kernel, padding='same', kernel_initializer='he_normal',
                              name='block2_conv1')(block1_pool)
        block2_bn1 = BatchNormalization(name='block2_bn1')(block2_conv1)
        block2_acti1 = Activation(activation='relu', name='block2_acti1')(block2_bn1)

        block2_conv2 = Conv2D(nb_filters * 2, kernel, padding='same', kernel_initializer='he_normal',
                              name='block2_conv2')(block2_acti1)
        block2_bn2 = BatchNormalization(name='block2_bn2')(block2_conv2)
        block2_acti2 = Activation(activation='relu', name='block2_acti2')(block2_bn2)

        block2_pool = MaxPool2D(pool_size=(2, 2), name='block2_pool')(block2_acti2)

        block3_conv1 = Conv2D(nb_filters * 4, kernel, padding='same', kernel_initializer='he_normal',
                              name='block3_conv1')(block2_pool)
        block3_bn1 = BatchNormalization(name='block3_bn1')(block3_conv1)
        block3_acti1 = Activation(activation='relu', name='block3_acti1')(block3_bn1)

        block3_conv2 = Conv2D(nb_filters * 4, kernel, padding='same', kernel_initializer='he_normal',
                              name='block3_conv2')(block3_acti1)
        block3_bn2 = BatchNormalization(name='block3_bn2')(block3_conv2)
        block3_acti2 = Activation(activation='relu', name='block3_acti2')(block3_bn2)

        block3_conv3 = Conv2D(nb_filters * 4, kernel, padding='same', kernel_initializer='he_normal',
                              name='block3_conv3')(block3_acti2)
        block3_bn3 = BatchNormalization(name='block3_bn3')(block3_conv3)
        block3_acti3 = Activation(activation='relu', name='block3_acti3')(block3_bn3)

        up_conv1 = Conv2D(nb_filters, kernel, activation='relu', padding='same', kernel_initializer='he_normal',
                          name='up_conv1')(block1_acti2)

        up1 = UpSampling2D(size=(2, 2), name='up1')(block2_acti2)
        up_conv2 = Conv2D(nb_filters, kernel, activation='relu', padding='same', kernel_initializer='he_normal',
                          name='up_conv2')(up1)

        up2 = UpSampling2D(size=(4, 4), name='up2')(block3_acti3)
        up_conv3 = Conv2D(nb_filters, kernel, activation='relu', padding='same', kernel_initializer='he_normal',
                          name='up_conv3')(up2)

        concat = Concatenate(axis=3)([up_conv1, up_conv2, up_conv3])
        output = Conv2D(4, 1, activation='softmax', kernel_initializer='he_normal',
                        name='output')(concat)

        model = Model(inputs=x, outputs=output)
        model.summary()
        return model

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

    def step_decay(self, epochs):
        # init_rate = 0.01
        # fin_rate = 0.001
        # total_epochs = 20
        if epochs < 30:
            lrate = 0.001
        elif epochs >= 30 and epochs < 50:
            lrate = 0.0001
            # lrate = init_rate - (init_rate - fin_rate) / total_epochs * float(epochs)
        else:
            lrate = 0.00003
        return lrate

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

        print(model.summary())
        change_lr = LearningRateScheduler(self.step_decay)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)

        sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        # model.load_weights('../weights/FCDense_tfcv4.22-0.0269.hdf5')
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer=adam, loss='categorical_crossentropy',
                      metrics=['accuracy', self.dice_comp, self.dice_core, self.dice_en])  # adam 0.001 SGD 0.0001 SGD 0.00001
        checkpointer = ModelCheckpoint(filepath='../weights/VGG/3/VGG16_tfcv5.{epoch:02d}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, epochs=70, validation_data=(X_test, Y_test), batch_size=16, shuffle=True,
                         callbacks=[checkpointer, change_lr])

        with open('../weights/VGG/3/VGG16_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)

    def test(self):
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        print(X_test.shape)
        model = self.get_model()
        model.load_weights('../weights/VGG/3/VGG16_tfcv5.70.hdf5')
        pred = model.predict(X_test, batch_size=4, verbose=1)
        # np.save('./VGG3/VGG16_tfcv5_5.npy', pred)
        print(pred.shape)
        print('#### start test ###')
        pred1 = np.argmax(pred, axis=3)
        print(pred1.shape)
        np.save('../results/VGG/3/vgg_test.npy', pred1)


class VGG4():
    def conv_block(self, x, block, nb_filters, kernel):
        base_name = 'block' + str(block) + '_'
        x = Conv2D(nb_filters, kernel, padding='same', activation='relu', name=base_name + 'conv1')(x)
        x = Conv2D(nb_filters, kernel, padding='same', activation='relu', name=base_name + 'conv2')(x)
        if (block > 2):
            x = Conv2D(nb_filters + 64, kernel, padding='same', activation='relu', name=base_name + 'conv3')(x)
            x = MaxPool2D(pool_size=(2, 2), name=base_name + 'pool')(x)
        else:
            x = MaxPool2D(pool_size=(2, 2), name=base_name + 'pool')(x)
        return x

    def get_model(self, block=5, nb_filters=64, kernel=3):
        input_tensor = Input((240, 240, 4))
        x = input_tensor
        # for each_block in range(1, block):
        #
        #     x = conv_block(x, each_block, nb_filters=nb_filters * each_block, kernel=kernel)
        #
        # model = Model(inputs=input_tensor, outputs=x)
        # model.summary()

        block1_conv1 = Conv2D(nb_filters, kernel, padding='same', kernel_initializer='he_normal',
                              name='block1_conv1')(x)
        block1_bn1 = BatchNormalization(name='block1_bn1')(block1_conv1)
        block1_acti1 = Activation(activation='relu', name='block1_acti1')(block1_bn1)

        block1_conv2 = Conv2D(nb_filters, kernel, padding='same', kernel_initializer='he_normal',
                              name='block1_conv2')(block1_acti1)
        block1_bn2 = BatchNormalization(name='block1_bn2')(block1_conv2)
        block1_acti2 = Activation(activation='relu', name='block1_acti2')(block1_bn2)

        block1_pool = MaxPool2D(pool_size=(2, 2), name='block1_pool')(block1_acti2)

        block2_conv1 = Conv2D(nb_filters * 2, kernel, padding='same', kernel_initializer='he_normal',
                              name='block2_conv1')(block1_pool)
        block2_bn1 = BatchNormalization(name='block2_bn1')(block2_conv1)
        block2_acti1 = Activation(activation='relu', name='block2_acti1')(block2_bn1)

        block2_conv2 = Conv2D(nb_filters * 2, kernel, padding='same', kernel_initializer='he_normal',
                              name='block2_conv2')(block2_acti1)
        block2_bn2 = BatchNormalization(name='block2_bn2')(block2_conv2)
        block2_acti2 = Activation(activation='relu', name='block2_acti2')(block2_bn2)

        block2_pool = MaxPool2D(pool_size=(2, 2), name='block2_pool')(block2_acti2)

        block3_conv1 = Conv2D(nb_filters * 4, kernel, padding='same', kernel_initializer='he_normal',
                              name='block3_conv1')(block2_pool)
        block3_bn1 = BatchNormalization(name='block3_bn1')(block3_conv1)
        block3_acti1 = Activation(activation='relu', name='block3_acti1')(block3_bn1)

        block3_conv2 = Conv2D(nb_filters * 4, kernel, padding='same', kernel_initializer='he_normal',
                              name='block3_conv2')(block3_acti1)
        block3_bn2 = BatchNormalization(name='block3_bn2')(block3_conv2)
        block3_acti2 = Activation(activation='relu', name='block3_acti2')(block3_bn2)

        block3_conv3 = Conv2D(nb_filters * 4, kernel, padding='same', kernel_initializer='he_normal',
                              name='block3_conv3')(block3_acti2)
        block3_bn3 = BatchNormalization(name='block3_bn3')(block3_conv3)
        block3_acti3 = Activation(activation='relu', name='block3_acti3')(block3_bn3)

        up_conv1 = Conv2D(nb_filters, kernel, activation='relu', padding='same', kernel_initializer='he_normal',
                          name='up_conv1')(block1_acti2)

        up1 = UpSampling2D(size=(2, 2), name='up1')(block2_acti2)
        up_conv2 = Conv2D(nb_filters, kernel, activation='relu', padding='same', kernel_initializer='he_normal',
                          name='up_conv2')(up1)

        up2 = UpSampling2D(size=(4, 4), name='up2')(block3_acti3)
        up_conv3 = Conv2D(nb_filters, kernel, activation='relu', padding='same', kernel_initializer='he_normal',
                          name='up_conv3')(up2)

        concat = Concatenate(axis=3)([up_conv1, up_conv2, up_conv3])
        output = Conv2D(4, 1, activation='softmax', kernel_initializer='he_normal',
                        name='output')(concat)

        model = Model(inputs=x, outputs=output)
        model.summary()
        return model

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

    def step_decay(self, epochs):
        # init_rate = 0.01
        # fin_rate = 0.001
        # total_epochs = 20
        if epochs < 30:
            lrate = 0.001
        elif epochs >= 30 and epochs < 50:
            lrate = 0.0001
            # lrate = init_rate - (init_rate - fin_rate) / total_epochs * float(epochs)
        else:
            lrate = 0.00003
        return lrate

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

        print(model.summary())
        change_lr = LearningRateScheduler(self.step_decay)
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)

        sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
        # model.load_weights('../weights/FCDense_tfcv4.22-0.0269.hdf5')
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(optimizer=adam, loss='categorical_crossentropy',
                      metrics=['accuracy', self.dice_comp, self.dice_core, self.dice_en])  # adam 0.001 SGD 0.0001 SGD 0.00001
        checkpointer = ModelCheckpoint(filepath='../weights/VGG/4/VGG16_tfcv5.{epoch:02d}.hdf5',
                                       monitor='val_loss', verbose=1)
        hist = model.fit(X_train, Y_train, epochs=70, validation_data=(X_test, Y_test), batch_size=16, shuffle=True,
                         callbacks=[checkpointer, change_lr])

        with open('../weights/VGG/4/VGG16_tfcv5.json', 'w') as f:
            json.dump(hist.history, f)

    def test(self):
        X_train, Y_train, X_test, Y_test = self.load_data('../data/', mode=False)
        print(X_test.shape)
        model = self.get_model()
        model.load_weights('../weights/VGG/4/VGG16_tfcv5.70.hdf5')
        pred = model.predict(X_test, batch_size=4, verbose=1)
        # np.save('./VGG4/VGG16_tfcv5_5.npy', pred)
        print(pred.shape)
        print('#### start test ###')
        pred1 = np.argmax(pred, axis=3)
        print(pred1.shape)
        np.save('../results/VGG/4/vgg_test.npy', pred1)






if __name__ == '__main__':
    VGG0 = VGG0()
    VGG0.test()

    VGG1 = VGG1()
    VGG1.test()

    VGG2 = VGG2()
    VGG2.test()

    VGG3 = VGG3()
    VGG3.test()

    VGG4 = VGG4()
    VGG4.test()


