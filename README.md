## 0. Environment setup
Please ensure your Python environment is consistent with one presented in `Pipfile`. You can do so by creating a virtual environment by invoking `pipenv install` or installing packages manually.

## 1. Visual Wake Words: Data preprocessing
Please set the `VISUALWAKEWORDS_CONFIG` dictionary (located at the top of `datasets.py`) to your data directories and 
JSON files build for the dataset. 

The preprocessing pipeline resizes images to 128x128 and applies various data augmentations for training. 

## 2. Model: downscaled EfficientNet-B0
We use an EfficientNet family of models [1], whose depth and width (number of channels in `Conv` outputs) are defined according to a formula (see the manuscript). We substitute lower values than those explored in the paper to obtain a model that has:
 * < 250KB peak memory usage (specifically, 229'376 B, assuming fused Conv-BN-ReLU blocks)
 * < 250KB parameters (215'040 parameters at 8 bits per parameter)
 * < 60 MB multiply-adds (specifically, 11'100'528).

The model is quantised to 8-bit precision, using a customised version of `tf.contrib.quantize`. 

## 3. Running the model
```
python quantizer.py --dataset visualwakewords --arch efficientnet_bz 
    --checkpoint-dir <some_directory> 
    --log-dir <some_directory> 
    --batch 256 --lr 0.03 --epochs 350
```

## 4. Results: MSCOCO minival

TFLite model: 83.12%

Pre-TFLite model with fake quantisation: 83.22%

See the `model` folder for serialised models and checkpoint information.

## Credit
Authors: Royson Lee*, Edgar Liberis*, Nic Lane (* --- eq. contrib.)

Our implementation incorporates the following open-source code:
* EfficientNet family implementation in Keras: https://github.com/qubvel/efficientnet
* AutoAugment augmentation policy: https://github.com/DeepVoltaire/AutoAugment

### References
[1] Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. arXiv preprint arXiv:1905.11946.
