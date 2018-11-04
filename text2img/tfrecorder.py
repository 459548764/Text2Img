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
SCALE = 299
# data type for each value of iterator
TOKEN_TYPE = np.uint32
LENGTH_TYPE = np.uint8
IMAGE_TYPE = np.uint8


# def shuffle_data(data, seed=0):
#     np.random.seed(seed)
#     np.random.shuffle(data)
#     return data


def text_statistics(list_of_sentence):
    length_token = 0
    length_char = 0
    dict_token = dict()
    dict_char = dict()
    dict_token[OOV_IDENTIFIER] = 0
    dict_char[OOC_IDENTIFIER] = 0

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

    return (length_token, dict_token), (length_char, dict_char)


def read_tfrecord(length_token: int,
                  length_char: int,
                  length_caption: int):
    """ Reader for TFRecord """

    def __read_tfrecord(example_proto):
        features = dict()
        features['image'] = tf.FixedLenFeature([], tf.string, default_value="")
        features['caption_word'] = tf.FixedLenFeature([], tf.string, default_value="")
        features['caption_char'] = tf.FixedLenFeature([], tf.string, default_value="")
        features['length_word'] = tf.FixedLenFeature([], tf.string, default_value="")
        features['length_caption'] = tf.FixedLenFeature([], tf.int64, default_value=0)
        parsed_features = tf.parse_single_example(example_proto, features)

        def decode(name, shape, cast_type=tf.float32, raw_type=tf.uint8):
            if raw_type == np.uint32:
                raw_type = np.int32
            tmp = parsed_features[name]
            tmp = tf.decode_raw(tmp, raw_type)
            tmp = tf.cast(tmp, cast_type)
            tmp = tf.reshape(tmp, shape)
            return tmp

        return_data = list()
        return_data.append(decode('image', [SCALE, SCALE, 3], cast_type=tf.int32, raw_type=IMAGE_TYPE))
        return_data.append(decode('caption_word', [length_caption, length_token], cast_type=tf.int32, raw_type=TOKEN_TYPE))
        return_data.append(decode('caption_char', [length_caption, length_token, length_char], cast_type=tf.int32, raw_type=TOKEN_TYPE))
        return_data.append(decode('length_word', [length_caption], cast_type=tf.int32, raw_type=LENGTH_TYPE))
        return_data.append(tf.cast(parsed_features['length_caption'], tf.int32))
        return tuple(return_data)

    return __read_tfrecord


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
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tostring()]))

    def int_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

    mkdir('/'.join(path_to_save.split('/')[:-1]))
    my_print('start processing: %s -> %s' % (path_to_image, path_to_save))

    image_filenames = sorted(glob('%s/*.jpg' % path_to_image))
    name = '%s.tfrecord' % path_to_save

    annotation_dict = json.load(open(path_to_annotation))  # key: ['info', 'images', 'licenses', 'annotations']
    # file_name -> image_id
    name_to_id = dict([(i['file_name'], i['id']) for i in annotation_dict['images']])
    # image_id -> list of captions
    id_to_caption = dict()
    docs = []
    for i in annotation_dict['annotations']:
        docs.append(i['caption'])
        if i['image_id'] in id_to_caption.keys():
            id_to_caption[i['image_id']] += [i['caption']]
        else:
            id_to_caption[i['image_id']] = [i['caption']]

    # statistics for text
    if os.path.exists(path_to_meta_dict):
        my_print('loading caption statistics....')
        meta_dict = json.load(open(path_to_meta_dict))
        length_token = meta_dict['length_token']
        dict_token = meta_dict['dict_token']
        length_char = meta_dict['length_char']
        dict_char = meta_dict['dict_char']
        length_caption = meta_dict['length_caption']
    else:
        my_print('aggregating statistics from captions....')

        length_caption = int(np.max([len(v) for v in id_to_caption.values()]))
        (length_token, dict_token), (length_char, dict_char) = text_statistics(docs)
        with open(path_to_meta_dict, 'w') as outfile:
            meta = dict(
                length_token=length_token,
                dict_token=dict_token,
                length_char=length_char,
                dict_char=dict_char,
                length_caption=length_caption)
            json.dump(meta, outfile)

    # dict_token_inv = dict([(v, k) for k, v in dict_token.items()])

    # create recorder
    full_size = len(image_filenames)
    my_print('writing tfrecord: size %i' % full_size)
    compress_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    with tf.python_io.TFRecordWriter(name, options=compress_opt) as writer:
        time_stamp = time()
        time_stamp_start = time()
        for n, single_image_file in enumerate(image_filenames):

            feature = dict()

            ######################
            # processing caption #
            ######################
            img_file_name = single_image_file.split('/')[-1]
            img_id = name_to_id[img_file_name]
            list_of_captions = id_to_caption[img_id]

            full_word_id = []
            full_sequence_length = []
            full_char_id = []

            for _n in range(length_caption):
                if _n < len(list_of_captions):
                    caption = list_of_captions[_n]
                    # print('\nraw :', caption)

                    tokens = caption.split()  # list of token
                    sequence_length = len(tokens)
                    if sequence_length >= length_token:
                        tokens = tokens[:length_token]
                        sequence_length = length_token

                    # get word id (length_token)
                    word_id = np.pad(
                        np.hstack([[dict_token[t] if t in dict_token.keys() else dict_token[OOV_IDENTIFIER]
                                    for t in tokens]]),
                        [0, length_token - sequence_length],
                        'constant')
                    # try:
                    #     word_id = np.pad(
                    #         np.hstack([
                    #             dict_token[t]
                    #             for t in tokens]),
                    #         [0, length_token - sequence_length],
                    #         'constant').astype(np.uint8)
                    # except KeyError as err:
                    # print(err)
                    # print('OOV', dict_token[OOV_IDENTIFIER], dict_token_inv[dict_token[OOV_IDENTIFIER]])
                    # print([dict_token[t] if t in dict_token.keys() else dict_token[OOV_IDENTIFIER]
                    #         for t in tokens])
                    # print(word_id[:sequence_length])
                    # print(word_id)
                    # print('conv:', ' '.join([dict_token_inv[i] for i in word_id[:sequence_length]]))

                    # get char id (length_token, length_char)
                    try:
                        raw_char_ids = np.vstack([
                            np.pad([
                                dict_char[c] if c in dict_char.keys() else dict_char[OOC_IDENTIFIER]
                                for c in t],
                                [0, length_char - len(t)],
                                'constant') for t in tokens
                        ])
                    except ValueError:
                        tmp = [
                            np.pad([
                                dict_char[c] if c in dict_char.keys() else dict_char[OOC_IDENTIFIER]
                                for c in t],
                                [0, int(np.max([length_char - len(t), 0]))],
                                'constant') for t in tokens
                        ]
                        raw_char_ids = np.vstack([t[:length_char] for t in tmp])
                    char_id = np.pad(
                        raw_char_ids,
                        [[0, length_token - sequence_length], [0, 0]],
                        'constant')

                else:
                    word_id = np.zeros([length_token])
                    sequence_length = 0
                    char_id = np.zeros([length_token, length_char])

                full_word_id.append(word_id)
                full_char_id.append(char_id)
                full_sequence_length.append(sequence_length)

            feature['caption_word'] = byte_feature(np.vstack(full_word_id).astype(TOKEN_TYPE))
            feature['caption_char'] = byte_feature(np.vstack(full_char_id).astype(TOKEN_TYPE))
            feature['length_word'] = byte_feature(np.hstack(full_sequence_length).astype(LENGTH_TYPE))
            feature['length_caption'] = int_feature(len(list_of_captions))

            ####################
            # processing image #
            ####################
            # get image instance corresponding to annotation's image id
            image_instance = Image.open(single_image_file)
            # resize image
            image_re = image_instance.resize((SCALE, SCALE))
            img = np.asarray(image_re)
            if img.shape != (SCALE, SCALE, 3):
                if img.shape == (SCALE, SCALE):  # if channel value is lacked, regard as gray scale
                    my_print('WARNING: find gray scale image: %s' % single_image_file)
                    img = np.tile(np.expand_dims(img, -1), 3)
                else:
                    my_print()
                    raise ValueError('Error: inconsistency shape', SCALE, img.shape)
            img = np.rint(img).clip(0, 255).astype(IMAGE_TYPE)
            feature['image'] = byte_feature(img)

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
