# Brain-Tumor-Segmentation
# Paper: Path aggregation U-Net model for brain tumor segmentation
Link:
https://link.springer.com/article/10.1007/s11042-020-08795-9

This repository contains the tensorflow implementation of the model we proposed in our paper

## Requirements

- The code has been written in Python (3.5.2) and Tensorflow (1.12.0)
- Make sure to install all the libraries given in requirement.txt (You can do so by the following command)
```
pip install -r requirement.txt
```


## Data preprocessing
[BraTS 2017](https://www.med.upenn.edu/sbia/brats2017.html) 
[BraTS 2018](https://www.med.upenn.edu/sbia/brats2018.html) 
* Has been finished

## Train (To get model)
* Select the GPU you want to use. Add the following at the beginning of training code.
```
>>> import os
>>> os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
>>> os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

* Begin training
* If you want to train all cross validation models. To run:
```
$ python3 ./train/FCNN.py
```
```
$ python3 ./train/DUnet.py
```
```
$ python3 ./train/VGG.py
```
```
$ python3 ./train/PA+FP.py
```
```
$ python3 ./train/PA+FP+ED.py
```
```
$ python3 ./train/PA+EFP+ED.py
```

* If you only want to train one model, To modify the code like following at the end of training code:
```
if __name__ == '__main__':
    paunet0 = PAUnet0()
    paunet0.train()

    # paunet1 = PAUnet1()
    # paunet1.train()
    #
    # paunet2 = PAUnet2()
    # paunet2.train()
    #
    # paunet3 = PAUnet3()
    # paunet3.train()
    #
    # paunet4 = PAUnet4()
    # paunet4.train()
```

## Test (To get the segmented .npy file)
* If you want to test all cross validation models. To run:
```
$ python3 ./test/FCNN_test.py
```
```
$ python3 ./test/DUnet_test.py
```
```
$ python3 ./test/VGG_test.py
```
```
$ python3 ./test/PA+FP_test.py
```
```
$ python3 ./test/PA+FP+ED_test.py
```
```
$ python3 ./test/PA+EFP+ED_test.py
```
* If you only want to test one model, To modify the code like following at the end of testing code:
```
if __name__ == '__main__':
    paunet0 = PAUnet0()
    paunet0.test()

    # paunet1 = PAUnet1()
    # paunet1.test()
    #
    # paunet2 = PAUnet2()
    # paunet2.test()
    #
    # paunet3 = PAUnet3()
    # paunet3.test()
    #
    # paunet4 = PAUnet4()
    # paunet4.test()
```
## Calculate Dice


* Modify the path and run:
```
$ python3 ./utils/read_json.py
```
* You can choose to calculate dice of cross validation or dice of one model in
```
./utils/read_json.py
```


## Calculate PPV and Sensitivity
* Modify the path and run:
```
$ python3 ./utils/cal_dice_ppv_sen_cross_validation.py
```
* You can also use:
```
./utils/cal_dice_ppv_sen_onemodel.py
```
to calculate the PPV and Sensitivity of one model




