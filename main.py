import os
from math import ceil
import cv2
import numpy as np
from keras_segmentation.models.fcn import fcn_8_vgg

from keras_segmentation.models.model_utils import transfer_weights
from keras_segmentation.models.pspnet import pspnet_50
from keras_segmentation.pretrained import pspnet_50_ADE_20K

from keras_segmentation.models.unet import vgg_unet, resnet50_unet, mobilenet_unet
from keras_segmentation.models.pspnet import vgg_pspnet, resnet50_pspnet
from keras_segmentation.models.segnet import resnet50_segnet
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.layers import Conv2D
from tensorflow.keras.callbacks import EarlyStopping


def combine_rgb_nir(base_dir, overwrite=False):
    dir_name = os.path.basename(base_dir)

    print(dir_name)

    rgb_dir = os.path.join(base_dir, "rgb", dir_name)
    nir_dir = os.path.join(base_dir, "nir", dir_name)
    rgb_nir_combined_dir = os.path.join(base_dir, "rgb_nir", dir_name)

    os.makedirs(rgb_nir_combined_dir, exist_ok=True)

    sorted_rgb_dir = sorted(os.listdir(rgb_dir))
    sorted_nir_dir = sorted(os.listdir(nir_dir))
    print(sorted_nir_dir)

    for rgb, nir in zip(sorted_rgb_dir, sorted_nir_dir):
        new_name = os.path.basename(rgb)
        # new_name = new_name.replace("png", 'tiff')
        new_name = os.path.join(rgb_nir_combined_dir, new_name)

        if overwrite or not os.path.exists(new_name):
            rgb = os.path.join(rgb_dir, rgb)
            nir = os.path.join(nir_dir, nir)

            print(new_name, rgb, nir)

            rgb = cv2.imread(rgb)
            nir = cv2.imread(nir)[:, :, -1]
            rows,cols = nir.shape

            M = np.float32([[1,0,-80],[0,1,-20]])
            nir = cv2.warpAffine(nir,M,(cols,rows))

            combined = np.dstack((rgb, nir))
            cv2.imwrite(new_name, combined)
        else:
            print("Skipping", new_name)

if __name__ == '__main__':
    #combine_rgb_nir('inputs/train', overwrite=True)
    #combine_rgb_nir('inputs/test', overwrite=True)
    #combine_rgb_nir('inputs/val', overwrite=True)


    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # pass
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    tf.random.set_seed(4256)


    #model = vgg_unet(n_classes=8,channels=4)
    model = vgg_pspnet(n_classes=8, channels=4)
    #model = resnet50_segnet(n_classes=8)
    #model = mobilenet_unet(n_classes=8)
    #model = fcn_8_vgg(n_classes=8, channels=4)
    #model = resnet50_pspnet(n_classes=8)

    callbacks = [
        EarlyStopping(patience=3, verbose=1)
    ]

    #model.summary()


    checkpoint_folder = f'checkpoints/{model.model_name}_nir'

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    print(model.model_name)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_folder)
    print(latest_checkpoint)

    if latest_checkpoint is not None:
        print("Weights loaded")
        model.load_weights(latest_checkpoint)


    batch_size = 2

    model.train(
        train_images='inputs/train/rgb_nir/train',
        train_annotations='inputs/train/mask/train',
        validate=True,
        val_images='inputs/val/rgb_nir/val',
        val_annotations='inputs/val/mask/val',
        checkpoints_path=f'{checkpoint_folder}/{model.model_name}_nir ',
        batch_size=batch_size,
        epochs=1,

        # load_weights=latest_checkpoint,

        ignore_zero_class=True,
        do_augment=True,

        augmentation_name="aug_geometric",
        #augmentation_name="aug_all",

        steps_per_epoch=int(ceil(159 / batch_size)) * 4,
        val_steps_per_epoch=int(ceil(28 / batch_size)),
        callbacks=callbacks
    )

    print(model.evaluate_segmentation(
        inp_images_dir='inputs/test/rgb_nir/test', annotations_dir='inputs/test/mask/test'
         , read_type=cv2.IMREAD_UNCHANGED
    ))
