import os
import json
import tensorflow as tf


class AttentionGAN:

    def __init__(self,
                 path_to_tfrecord: str,
                 path_to_meta: str,
                 batch: int = 1,
                 n_thread: int = 1):

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

        self.__embedding_size = embedding_size
        self.__base_batch = batch
        self.__path_to_tfrecord = path_to_tfrecord
        self.__n_thread = n_thread
        self.__build_graph()
        self.session = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.session.run(tf.global_variables_initializer())

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
        input_image = tf.cast(self.image, tf.float32)/255*2 - 1  # range -> [-1, 1]

        # word embedding
        with tf.device("/cpu:0"):
            with tf.variable_scope("word_embedding"):
                vocab_size = len(self.__dict_token)
                # mask the weight of `out of vocabulary index (padding as well)` as 0
                oov_index = tf.cast(0, tf.int32)
                full_index = tf.expand_dims(tf.range(vocab_size, dtype=tf.int32), -1)
                mask = tf.cast(tf.equal(full_index, oov_index), tf.float32)
                # embedding matrix
                var = tf.get_variable("embedding_matrix", [vocab_size, self.__embedding_size])
                var = var * mask
                emb_word = tf.nn.embedding_lookup(var, self.caption_word)

        # self.__generator()

        # tf.image.resize_images

    def __generator(self,
                    seed_value,
                    condition_feature,
                    word_features):
        """

        :param seed_value: random seed value
        :param condition_feature: conditional feature from sentence embedding
        :param word_features: word features
        :return:
        """


