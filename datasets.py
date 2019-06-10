import numpy as np
from tensorpack.dataflow import dataset, MapData, imgaug, AugmentImageComponent, PrefetchDataZMQ, MultiThreadMapData, \
    RNGDataFlow
import cv2
import json
import os
import multiprocessing

from autoaugment import ImageNetPolicy
from PIL import Image


VISUALWAKEWORDS_CONFIG = {
    "train_instances": "/datasets/COCO/2014/records/instances_visualwakewords_train2014.json",
    "train_images": "/datasets/COCO/2014/train2014",
    "val_instances": "/datasets/COCO/2014/records/instances_visualwakewords_val2014.json",
    "val_images": "/datasets/COCO/2014/val2014",
    "filter_list": "/datasets/COCO/2014/records/mscoco_minival_ids.txt"
}


def get_input_cifar10():
    train, test = dataset.Cifar10('train'), dataset.Cifar10('test', shuffle=False)

    def preprocess(x):
        image, label = x
        onehot = np.zeros(10)
        onehot[label] = 1.0
        return image, onehot

    return MapData(train, preprocess), MapData(test, preprocess), ((32, 32, 3), (10,))


def get_input_mnist():
    train, test = dataset.Mnist('train'), dataset.Mnist('test', shuffle=False)

    def preprocess(x):
        image, label = x
        image = np.expand_dims(image, axis=-1)  # Add a channels dimension
        onehot = np.zeros(10)
        onehot[label] = 1.0
        return image, onehot

    return MapData(train, preprocess), MapData(test, preprocess), ((28, 28, 1), (10,))


def get_input_imagenet():
    train = dataset.ILSVRC12("/datasets/ImageNet/ILSVRC/Data/CLS-LOC", "train", dir_structure="train", shuffle=True)
    test = dataset.ILSVRC12("/datasets/ImageNet/ILSVRC/Data/CLS-LOC", "val", dir_structure="train", shuffle=False)

    # Copied from tensorpack examples:
    # https://github.com/tensorpack/tensorpack/blob/master/examples/ImageNetModels/imagenet_utils.py

    train_augmentors = imgaug.AugmentorList([
        imgaug.GoogleNetRandomCropAndResize(interp=cv2.INTER_CUBIC),
        # It's OK to remove the following augs if your CPU is not fast enough.
        # Removing brightness/contrast/saturation does not have a significant effect on accuracy.
        # Removing lighting leads to a tiny drop in accuracy.
        imgaug.RandomOrderAug(
            [imgaug.BrightnessScale((0.6, 1.4), clip=False),
             imgaug.Contrast((0.6, 1.4), rgb=False, clip=False),
             imgaug.Saturation(0.4, rgb=False),
             # rgb-bgr conversion for the constants copied from fb.resnet.torch
             imgaug.Lighting(0.1,
                             eigval=np.asarray(
                                 [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                             eigvec=np.array(
                                 [[-0.5675, 0.7192, 0.4009],
                                  [-0.5808, -0.0045, -0.8140],
                                  [-0.5836, -0.6948, 0.4203]],
                                 dtype='float32')[::-1, ::-1]
                             )]),
        imgaug.Flip(horiz=True),
    ])

    test_augmentors = imgaug.AugmentorList([
        imgaug.ResizeShortestEdge(256, interp=cv2.INTER_CUBIC),
        imgaug.CenterCrop((224, 224)),
    ])

    def preprocess(augmentors):
        def apply(x):
            image, label = x
            onehot = np.zeros(1000)
            onehot[label] = 1.0
            image = augmentors.augment(image)
            return image, onehot
        return apply

    parallel = min(40, multiprocessing.cpu_count() // 2)  # assuming hyperthreading
    train = MapData(train, preprocess(train_augmentors))
    train = PrefetchDataZMQ(train, parallel)

    test = MultiThreadMapData(test, parallel, preprocess(test_augmentors), strict=True)
    test = PrefetchDataZMQ(test, 1)

    return train, test, ((224, 224, 3), (1000,))


class VisualWakeWordsFiles(RNGDataFlow):
    def __init__(self, instances_file, img_folder, shuffle=False, filter_list=None):
        assert os.path.exists(instances_file)
        assert os.path.exists(img_folder) and os.path.isdir(img_folder)

        with open(instances_file, "r") as f:
            metadata = json.load(f)
        annotations = lambda img_id: metadata["annotations"][str(img_id)]
        keep_ids = None
        if filter_list is not None:
            with open(filter_list, "r") as f:
                keep_ids = set(int(line) for line in f)
        self.img_list = [(entry["file_name"],
                          int(any(d['label'] == 1 for d in annotations(entry["id"]))))
                         for entry in metadata["images"]
                         if keep_ids is None or entry["id"] in keep_ids]
        self.base_path = img_folder
        self.shuffle = shuffle

        if filter_list:
            assert len(keep_ids) == len(self.img_list)

    def __len__(self):
        return len(self.img_list)

    def __iter__(self):
        idxs = np.arange(len(self.img_list))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, label = self.img_list[k]
            fname = os.path.join(self.base_path, fname)
            yield [fname, label]


class VisualWakeWords(VisualWakeWordsFiles):
    def __init__(self, instances_file, img_folder, shuffle=False, filter_list=None):
        super().__init__(instances_file, img_folder, shuffle, filter_list)

    def __iter__(self):
        for fname, label in super().__iter__():
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            assert im is not None, fname
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            yield [im, label]


def get_input_visualwakewords():
    # VISUALWAKEWORDS_CONFIG = {
    #     "train_instances": "/home/el/Datasets/COCO14/visualwakewords/instances_visualwakewords_train2014.json",
    #     "train_images": "/home/el/Datasets/COCO14/train2014",
    #     "val_instances": "/home/el/Datasets/COCO14/visualwakewords/instances_visualwakewords_val2014.json",
    #     "val_images": "/home/el/Datasets/COCO14/val2014",
    #     "filter_list": "/home/el/Datasets/COCO14/visualwakewords/mscoco_minival_ids.txt"
    # }

    target_side_size = 128
    train = VisualWakeWords(VISUALWAKEWORDS_CONFIG["train_instances"], VISUALWAKEWORDS_CONFIG["train_images"], shuffle=True)
    test = VisualWakeWords(VISUALWAKEWORDS_CONFIG["val_instances"], VISUALWAKEWORDS_CONFIG["val_images"],
                           filter_list=VISUALWAKEWORDS_CONFIG["filter_list"], shuffle=False)

    train_augmentors = imgaug.AugmentorList([
        imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(0.25, 1.), target_shape=target_side_size,
                                            interp=cv2.INTER_CUBIC),
        imgaug.Flip(horiz=True),
    ])

    autoaugment_policy = ImageNetPolicy()

    test_augmentors = imgaug.AugmentorList([
        imgaug.ResizeShortestEdge(int(target_side_size * 1.2), interp=cv2.INTER_CUBIC),
        imgaug.CenterCrop((target_side_size, target_side_size)),
    ])

    def preprocess(train):
        def apply(x):
            image, label = x
            onehot = np.zeros(2)
            onehot[label] = 1.0
            augmentors = train_augmentors if train else test_augmentors
            image = augmentors.augment(image)
            if train:
                image = np.array(autoaugment_policy(Image.fromarray(image)))
            return image, onehot
            # mean = [0.4767, 0.4488, 0.4074]
            # std = [0.2363, 0.2313, 0.2330]
            # return (image / 255.0 - mean) / std, onehot
        return apply


    parallel = min(18, multiprocessing.cpu_count() // 2)  # assuming hyperthreading
    train = MapData(train, preprocess(train=True))
    train = PrefetchDataZMQ(train, parallel)

    test = MultiThreadMapData(test, parallel, preprocess(train=False), strict=True)
    test = PrefetchDataZMQ(test, 1)

    return train, test, ((target_side_size, target_side_size, 3), (2,))


DATASETS = {
    "MNIST": get_input_mnist,
    "CIFAR10": get_input_cifar10,
    "imagenet": get_input_imagenet,
    "visualwakewords": get_input_visualwakewords
}
