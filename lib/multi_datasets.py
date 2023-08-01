import sys
import random
import functools
import tensorflow as tf
import numpy as np
import os
import json

from lib.utils import load_json

SHUFFLE_BUFFER_SIZE = 1000
NUM_PARALLEL_CALLS = tf.data.experimental.AUTOTUNE
PREFETCH_BUFFER_SIZE = tf.data.experimental.AUTOTUNE

def build_dataset_multi2(root, path, batch_size, random_bright=False, training=True):
    def make_addtional_function_ans(image, label):
        gray_a = tf.reduce_mean(tf.image.rgb_to_grayscale(image))
        gray_b = tf.reduce_mean(tf.image.rgb_to_grayscale(label))

        alpha = 0

        if gray_a > gray_b:
            alpha = 1
        return alpha

    def transform_train(image, label, root):
        image1 = read_image_origin(tf.strings.join([root, image], '/'))
        label1 = read_image_origin(tf.strings.join([root, label], '/'))

        image_train_list = json.load(open(path, "r"))['image']
        image_label_list = json.load(open(path, "r"))['label']

        image_train_list = tf.expand_dims(image_train_list, axis=-1)
        image_label_list = tf.expand_dims(image_label_list, axis=-1)

        concat_list = tf.concat([image_train_list, image_label_list], axis=1)
        r_idx = tf.random.shuffle(concat_list)[0]

        r_input = r_idx[0]
        r_label = r_idx[1]

        image2 = read_image_origin(tf.strings.join([root, r_input], '/'))
        label2 = read_image_origin(tf.strings.join([root, r_label], '/'))

        if random_bright:
            image1, label1 = random_crop_and_brightness(image1, label1)
            image2, label2 = random_crop_and_brightness(image2, label2)
        else:
            image1, label1 = random_crop_and_resize_v3(image1, label1)
            image2, label2 = random_crop_and_resize_v3(image2, label2)

        image1, label1 = random_rotate_and_flip(image1, label1)
        image2, label2 = random_rotate_and_flip(image2, label2)

        image1 = normalize_image(image1)
        label1 = normalize_image(label1)
        image2 = normalize_image(image2)
        label2 = normalize_image(label2)

        alpha = make_addtional_function_ans(image1, image2)

        x = list()
        y = list()
        x.append([image1, image2])
        y.append([label1, label2])
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)

        return x, y, alpha

    def transform_test(image, label, root):
        image1 = read_image_origin(tf.strings.join([root, image], '/'))
        label1 = read_image_origin(tf.strings.join([root, label], '/'))

        image1 = normalize_image(image1)
        label1 = normalize_image(label1)

        return image1, label1

    def read_image_origin(path):
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.convert_image_dtype(image, dtype='float32')
        return image

    def random_crop_and_resize_v3(image, label):
        x = tf.concat([image, label], axis=-1)

        h = tf.shape(image)[0]
        w = tf.shape(image)[1]
        if tf.get_static_value(h) != None:
            h_r = h//2
            w_r = w//2
            if h_r > 512 and w_r > 512:
                h = random.randint(256,h_r)
                w = random.randint(256,w_r)
        x = tf.image.random_crop(x, [h, w, 6])
        x = tf.image.resize(x, [384, 384])
        image, label = tf.split(x, [3, 3], axis=-1)
        return image, label

    def normalize_image(image):
        image = tf.clip_by_value(image, 0.0, 1.0)
        image = 2.0 * image - 1.0
        return image

    db = load_json(path)
    ds = tf.data.Dataset.from_tensor_slices((db['image'], db['label']))
    if training:
        ds = ds.shuffle(SHUFFLE_BUFFER_SIZE)
        transform = functools.partial(transform_train, root=root)
    else:
        transform = functools.partial(transform_test, root=root)
    ds = ds.map(transform, num_parallel_calls=NUM_PARALLEL_CALLS)
    ds = ds.batch(batch_size)
    return ds

def read_image_v2(path):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, dtype='float32')
    image = tf.image.resize(image, [512, 512])
    return image

def random_crop_and_resize_v2(image, label):
    x = tf.concat([image, label], axis=-1)
    h = random.randint(256, 512)
    w = random.randint(256, 512)
    x = tf.image.random_crop(x, [h, w, 6])
    x = tf.image.resize(x, [256, 256])
    image, label = tf.split(x, [3, 3], axis=-1)
    return image, label

def normalize_image(image):
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = 2.0 * image - 1.0
    return image

def read_image(path, resize=False):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype='float32')
    image = 2.0 * image - 1.0
    image.set_shape((None, None, 3))

    return image

def random_rotate_and_flip(image, label):
    ra = tf.random.uniform((), 0, 4, dtype='int32')
    x = tf.concat([image, label], axis=-1)
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.rot90(x, ra)
    image, label = tf.split(x, [3, 3], axis=-1)
    return image, label

def random_crop_and_brightness(image, label):
    x = tf.image.random_brightntess(image,max_delta=0.2)
    x = tf.clip_by_value(image1, -1.0, 1.0)
    x = tf.concat([image, label], axis=-1)
    x = tf.image.resize(x, [512, 512])
    h = random.randint(256, 512)
    w = random.randint(256, 512)
    x = tf.image.random_crop(x, [h, w, 6])
    x = tf.image.resize(x, [256, 256])
    image, label = tf.split(x, [3, 3], axis=-1)
    return image, label

def random_color_distortion(image, label):
    rv = random.uniform(-0.25, 0.25)
    image = (1.0 - rv) * image + rv * label
    return image

