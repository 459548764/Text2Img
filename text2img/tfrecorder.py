from glob import glob
from time import time
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
from .util import mkdir

OOV_IDENTIFIER = '####OOV####'  # out of vocabulary
OOC_IDENTIFIER = '####OOC####'  # out of character
SCALE = [64, 128, 256, 299]


def shuffle_data(data, seed=0):
    np.random.seed(seed)
    np.random.shuffle(data)
    return data


def tfrecord_parser(img_shape):
    def __tfrecord_parser(example_proto):
        features = dict(image=tf.FixedLenFeature([], tf.string, default_value=""))
        parsed_features = tf.parse_single_example(example_proto, features)
        feature_image = tf.decode_raw(parsed_features["image"], tf.uint8)
        feature_image = tf.cast(feature_image, tf.float32)
        image = tf.reshape(feature_image, img_shape)
        return image
    return __tfrecord_parser


def text_statistics(list_of_sentence):
    length_token = 0
    length_char = 0
    dict_token = dict()
    dict_char = dict()

    for sentence in list_of_sentence:
        tokens = sentence.split()
        length_token = int(np.max([length_token, len(tokens)]))
        for single_token in tokens:
            length_char = int(np.max([length_char, len(single_token)]))
            if single_token not in dict_token.keys():
                dict_token[single_token] = len(dict_token.keys())
            for char in single_token:
                if char not in dict_char.keys():
                    dict_char[char] = len(dict_char.keys())

    dict_token[OOV_IDENTIFIER] = len(dict_token.keys())
    dict_char[OOC_IDENTIFIER] = len(dict_char.keys())
    return (length_token, dict_token), (length_char, dict_char)


def read_tfrecord():

    def __tfrecord_parser(example_proto):
        features = dict()
        for scale in SCALE:
            features['image_%i' % scale] = tf.FixedLenFeature([], tf.string, default_value="")
        features['caption_word'] = tf.FixedLenFeature([], tf.string, default_value="")
        features['caption_char'] = tf.FixedLenFeature([], tf.string, default_value="")
        features['length'] = tf.FixedLenFeature([], tf.string, default_value="")

        parsed_features = tf.parse_single_example(example_proto, features)
        feature_image = tf.decode_raw(parsed_features["image"], tf.uint8)
        feature_image = tf.cast(feature_image, tf.float32)

        image = tf.reshape(feature_image, img_shape)
        return image
    return __tfrecord_parser


def create_tfrecord(path_to_image: str,
                    path_to_annotation: str,
                    path_to_save: str,
                    path_to_meta_dict: str,
                    print_progress: bool = True,
                    progress_interval: int = 10):

    """ Formatting data as TFrecord """

    def my_print(*args, **kwargs):
        if print_progress:
            print(*args, **kwargs)

    def byte_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.astype(float).tostring()]))

    def int_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

    mkdir('/'.join(path_to_save.split('/')[:-1]))
    my_print('start processing: %s -> %s' % (path_to_image, path_to_save))

    image_filenames = sorted(glob('%s/*.jpg' % path_to_image))
    annotation_dict = json.load(open(path_to_annotation))  # key: ['info', 'images', 'licenses', 'annotations']
    name = '%s.tfrecord' % path_to_save
    # image_id -> file_name
    id_to_name = dict([(i['id'], i['file_name']) for i in annotation_dict['images']])
    # statistics for text
    docs = [i['caption'] for i in annotation_dict['annotations']]
    if os.path.exists(path_to_meta_dict):
        my_print('loading caption statistics....')
        meta_dict = json.load(open(path_to_meta_dict))
        length_token = meta_dict['length_token']
        dict_token = meta_dict['dict_token']
        length_char = meta_dict['length_char']
        dict_char = meta_dict['dict_char']
    else:
        my_print('aggregating statistics from captions....')
        (length_token, dict_token), (length_char, dict_char) = text_statistics(docs)
        with open(path_to_meta_dict, 'w') as outfile:
            meta = dict(
                length_token=length_token,
                dict_token=dict_token,
                length_char=length_char,
                dict_char=dict_char)
            json.dump(meta, outfile)

    # create recorder
    full_size = len(image_filenames)
    my_print('writing tfrecord: size %i' % full_size)
    compress_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    with tf.python_io.TFRecordWriter(name, options=compress_opt) as writer:
        time_stamp = time()
        time_stamp_start = time()
        for n, single_annotation_dict in enumerate(annotation_dict['annotations']):  # ['image_id', 'id', 'caption']

            feature = dict()

            #########################
            # processing annotation #
            #########################
            annotation = single_annotation_dict['caption']
            tokens = annotation.split()  # list of token
            sequence_length = len(tokens)
            # get word id (length_token)
            word_id = np.pad(
                np.hstack([
                    dict_token[t] if t in dict_token.keys() else dict_token[OOV_IDENTIFIER]
                    for t in tokens]),
                [0, length_token - sequence_length],
                'constant')
            feature['caption_word'] = byte_feature(word_id)
            feature['length'] = int_feature(sequence_length)

            # get char id (length_token, length_char)
            raw_char_ids = np.vstack([
                np.pad([
                    dict_char[c] if c in dict_char.keys() else dict_char[OOC_IDENTIFIER]
                    for c in t],
                    [0, length_char - len(t)], 'constant') for t in tokens
            ])
            char_id = np.pad(
                raw_char_ids,
                [0, length_token - sequence_length],
                'constant')
            feature['caption_char'] = byte_feature(char_id)

            ####################
            # processing image #
            ####################
            # get image instance corresponding to annotation's image id
            image_id = single_annotation_dict['image_id']
            file_name = id_to_name[image_id]
            image_instance = Image.open('%s/%s' % (path_to_image, file_name))
            # resize image
            for resize_val in SCALE:
                image_re = image_instance.resize((resize_val, resize_val))
                img = np.asarray(image_re)
                if img.shape != (resize_val, resize_val, 3):
                    if img.shape != (resize_val, resize_val):
                        my_print('WARNING: find gray scale image: %s' % file_name)
                        img = np.tile(np.expand_dims(img, -1), 3)
                    else:
                        my_print()
                        raise ValueError('Error: inconsistency shape', resize_val, img.shape)
                img = np.rint(img).clip(0, 255).astype(np.uint8)
                feature['image_%i' % resize_val] = byte_feature(img)

            # display progress
            if n % progress_interval == 0:
                progress_perc = n / full_size * 100
                cl_time = time() - time_stamp
                whole_time = time() - time_stamp_start
                time_per_sam = cl_time / progress_interval
                my_print(
                    '%d / %d (%0.1f %%), %0.4f sec/image (%0.1f sec) \r'
                    % (n, full_size, progress_perc, time_per_sam, whole_time), end='', flush=True)
                time_stamp = time()

            # write record
            ex = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(ex.SerializeToString())