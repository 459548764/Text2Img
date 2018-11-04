import tensorflow as tf
import os
import json
import numpy as np
from .tfrecorder import read_tfrecord


class SampleModel:

    def __init__(self,
                 path_to_tfrecord: str,
                 path_to_meta: str,
                 batch: int=1,
                 n_thread: int=1):

        if os.path.exists(path_to_meta):
            meta_dict = json.load(open(path_to_meta))
            self.__length_token = meta_dict['length_token']
            self.__dict_token = meta_dict['dict_token']
            self.__dict_token_inv = dict([(v, k) for k, v in self.__dict_token.items()])
            self.__length_char = meta_dict['length_char']
            self.__dict_char = meta_dict['dict_char']
            self.__dict_char_inv = dict([(v, k) for k, v in self.__dict_char.items()])
            self.__length_caption = meta_dict['length_caption']
        else:
            raise ValueError('No meta file found.: %s' % path_to_meta)

        self.__base_batch = batch
        self.__path_to_tfrecord = path_to_tfrecord
        self.__n_thread = n_thread
        self.__build_graph()
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.session.run(tf.global_variables_initializer())
        self.__ini_flg = False

    def __iterator(self, batch_size):
        # load tfrecord instance
        self.tfrecord_name = tf.placeholder(tf.string, name='tfrecord_dataset_name')
        data_set_api = tf.data.TFRecordDataset(self.tfrecord_name, compression_type='GZIP')
        # convert record to tensor
        data_set_api = data_set_api.map(read_tfrecord(length_token=self.__length_token,
                                                      length_char=self.__length_char,
                                                      length_caption=self.__length_caption),
                                        self.__n_thread)
        # set batch size
        data_set_api = data_set_api.shuffle(buffer_size=1000)
        data_set_api = data_set_api.batch(batch_size)
        # make iterator
        iterator = tf.data.Iterator.from_structure(data_set_api.output_types, data_set_api.output_shapes)
        ini_iterator = iterator.make_initializer(data_set_api)
        return iterator, ini_iterator

    def __build_graph(self):
        self.batch_size = tf.cast(
            tf.placeholder_with_default(self.__base_batch, [], name='batch_size'),
            tf.int64)
        iterator, self.ini_iterator = self.__iterator(self.batch_size)
        self.image, self.caption_word, self.caption_char, self.length_word, self.length_caption = iterator.get_next()

    def initialization(self):
        self.__ini_flg = True
        self.session.run(self.ini_iterator, feed_dict={self.tfrecord_name: self.__path_to_tfrecord})

    def get_data(self):
        if not self.__ini_flg:
            self.initialization()
        try:
            image, caption_word, caption_char, length_word, length_caption = self.session.run(
                [self.image, self.caption_word, self.caption_char, self.length_word, self.length_caption]
            )
        except tf.errors.OutOfRangeError:
            self.initialization()
            image, caption_word, caption_char, length_word, length_caption = self.session.run(
                [self.image, self.caption_word, self.caption_char, self.length_word, self.length_caption]
            )

        image, caption_word, caption_char, length_word, length_caption = \
            image[0], caption_word[0], caption_char[0], length_word[0], length_caption[0]
        image = image.astype(np.uint8)

        captions = []
        for i in range(length_caption):
            captions.append(' '.join([self.__dict_token_inv[_i] for _i in caption_word[i][:length_word[i]]]))

        return image, captions



