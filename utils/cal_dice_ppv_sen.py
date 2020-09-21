import numpy as np
import matplotlib.pyplot as plt
import json

# ppv sen
def PPV_whole(pred, true):
    TP = len(pred[((pred == 1) | (pred == 2) | (pred == 3)) & (pred == true)])
    FP = len(pred[((pred == 1) | (pred == 2) | (pred == 3)) & (pred != true)])
    FN = len(pred[(pred == 0) & (pred != true)])
    TN = len(pred[(pred == 0) & (pred == true)])
    print('TP=', TP, 'FP=', FP, 'FN=', FN, 'TN=', TN, 'sum=', sum((TP, FP, FN, TN)))
    if TP == 0:
        return 0, 0, 0
    return 2*TP / float(FN+2*TP+FP), TP / float(TP+FP), TP / float(TP+FN)


def PPV_core(pred, true):
    TP = len(pred[ ((pred==1) | (pred==3)) & (pred==true) ] )
    FP = len(pred[ ((pred==1) | (pred==3)) & (pred!=true) ] )
    FN = len(pred[ ((pred==0) | (pred==2)) & (pred!=true)])
    TN = len(pred[ ((pred==0) | (pred==2)) & (pred==true)])
    print(TP, FP, FN, TN, sum((TP, FP, FN, TN)))
    if TP == 0:
        return 0, 0, 0
    return 2*TP / float(FN+2*TP+FP), TP / float(TP+FP), TP / float(TP+FN)


def PPV_ET(pred, true):
    TP = len(pred[ (pred==3) & (pred==true) ] )
    FP = len(pred[ (pred==3) & (pred!=true) ] )
    FN = len(pred[ ((pred==0) | (pred==1) | (pred==2)) & (pred!=true)])
    TN = len(pred[ ((pred==0) | (pred==1) | (pred==2)) & (pred==true)])
    # print(TP, FP, FN, TN, sum((TP, FP, FN, TN)))
    if TP == 0:
        return 0, 0, 0
    return 2*TP / float(FN+2*TP+FP), TP / float(TP+FP), TP / float(TP+FN)


if __name__ == '__main__':

    avg_dice_whole = []
    avg_ppv_whole = []
    avg_sen_whole = []
    avg_dice_core = []
    avg_ppv_core = []
    avg_sen_core = []
    avg_dice_en = []
    avg_ppv_en = []
    avg_sen_en = []

    for i in [0, 1, 2, 3, 4]:
        pred_path = '../results/VGG/' + str(i) + '/vgg_test.npy'
        true_path = '../data/Y' + str(i) + '.npy'
        pred = np.load(pred_path)
        true = np.load(true_path)
        y_pred = np.squeeze(pred)
        y_true = np.argmax(true, axis=3)
        dice_whole, ppv_whole, sen_whole = PPV_whole(y_pred, y_true)
        dice_core, ppv_core, sen_core = PPV_core(y_pred, y_true)
        dice_en, ppv_en, sen_en = PPV_ET(y_pred, y_true)


        avg_dice_whole.append(dice_whole)
        avg_ppv_whole.append(ppv_whole)
        avg_sen_whole.append(sen_whole)
        avg_dice_core.append(dice_core)
        avg_ppv_core.append(ppv_core)
        avg_sen_core.append(sen_core)
        avg_dice_en.append(dice_en)
        avg_ppv_en.append(ppv_en)
        avg_sen_en.append(sen_en)

    final_avg_dice_whole = np.sum(avg_dice_whole) / len(avg_dice_whole)
    final_avg_ppv_whole = np.sum(avg_ppv_whole) / len(avg_ppv_whole)
    final_avg_sen_whole = np.sum(avg_sen_whole) / len(avg_sen_whole)
    final_avg_dice_core = np.sum(avg_dice_core) / len(avg_dice_core)
    final_avg_ppv_core = np.sum(avg_ppv_core) / len(avg_ppv_core)
    final_avg_sen_core = np.sum(avg_sen_core) / len(avg_sen_core)
    final_avg_dice_en = np.sum(avg_dice_en) / len(avg_dice_en)
    final_avg_ppv_en = np.sum(avg_ppv_en) / len(avg_ppv_en)
    final_avg_sen_en = np.sum(avg_sen_en) / len(avg_sen_en)

    print('dice_whole =', round(final_avg_dice_whole, 4), '\n',
          'dice_core =', round(final_avg_dice_core, 4), '\n',
          'dice_en =', round(final_avg_dice_en, 4), '\n', '\n',
          'ppv_whole =', round(final_avg_ppv_whole, 4), '\n',
          'ppv_core =', round(final_avg_ppv_core, 4), '\n',
          'ppv_en =', round(final_avg_ppv_en, 4), '\n',
          'sen_whole =', round(final_avg_sen_whole, 4), '\n',
          'sen_core =', round(final_avg_sen_core, 4), '\n',
          'sen_en =', round(final_avg_sen_en, 4))


