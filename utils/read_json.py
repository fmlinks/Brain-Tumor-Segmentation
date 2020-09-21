import json
import numpy as np


cross_validation = True


if not cross_validation:
    with open('../weights/PA+FP/4/PA+FP_tfcv5.json', 'r') as load_f:
        load_dict = json.load(load_f)

    val_dice_comp = load_dict['val_dice_comp'][69]
    val_dice_core = load_dict['val_dice_core'][69]
    val_dice_en = load_dict['val_dice_en'][69]
    print(val_dice_comp, val_dice_core, val_dice_en)
    # val_dice_comp = load_dict['val_dice_comp'][69]
    #
    # print(val_dice_comp)


if cross_validation:
    val_dice_comp = []
    val_dice_core = []
    val_dice_en = []

    for i in [0, 1, 2, 3, 4]:
        path = '../weights/PA+EFP+ED/' + str(i) + '/dunet_tfcv5.json'
        with open(path, 'r') as load_f:
            load_dict = json.load(load_f)


        val_dice_comp.append(load_dict['val_dice_comp'][69])
        val_dice_core.append(load_dict['val_dice_core'][69])
        val_dice_en.append(load_dict['val_dice_en'][69])

    avg_dice_comp = np.sum(val_dice_comp) / len(val_dice_comp)
    avg_dice_core = np.sum(val_dice_core) / len(val_dice_core)
    avg_dice_en = np.sum(val_dice_en) / len(val_dice_en)

    print(round(avg_dice_comp, 4), round(avg_dice_core, 4), round(avg_dice_en, 4))













