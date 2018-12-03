import os
import json
import tensorflow as tf
from . import util_tf


class AttentionGAN:

    def __init__(self,
                 ca_n,
                 seed_n,
                 embedding_size,
                 n_hidden_text_encoder,
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

        self.__ca_n = ca_n
        self.__seed_n = seed_n
        self.__embedding_size = embedding_size
        self.__n_hidden_text_enc = n_hidden_text_encoder
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
        __batch = util_tf.dynamic_batch_size(self.image)

        #################
        # process image #
        #################
        input_image = tf.cast(self.image, tf.float32) / 255 * 2 - 1  # range -> [-1, 1]
        input_image_0 = tf.image.resize_images(input_image, [64, 64, 3])
        input_image_1 = tf.image.resize_images(input_image, [128, 128, 3])
        input_image_2 = tf.image.resize_images(input_image, [256, 256, 3])

        ###################
        # process caption #
        ###################

        # choose one caption for each image
        def __sample_caption_id(batch_id):
            cap_len = self.length_caption[batch_id]
            ind = tf.random_uniform([], minval=0, maxval=cap_len, dtype=tf.int32)
            return ind

        caption_ids = tf.stack(tf.map_fn(__sample_caption_id, tf.range(__batch), dtype=tf.int32))

        def __sample_caption(batch_id):
            _id = caption_ids[batch_id]
            single_caption = self.caption_word[batch_id][_id]
            return single_caption

        def __sample_caption_length(batch_id):
            _id = caption_ids[batch_id]
            cap_len = self.length_caption[batch_id][_id]
            return cap_len

        ##############
        # main model #
        ##############
        # batch, max_token_length
        caption = tf.stack(tf.map_fn(__sample_caption, tf.range(__batch), dtype=tf.int32))
        caption_length = tf.stack(tf.map_fn(__sample_caption_length, tf.range(__batch), dtype=tf.int32))

        # random embedding
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
                emb_word = tf.nn.embedding_lookup(var, caption)

        # encoding text
        word_embedding, sentence_embedding = self.__text_encoder(token=emb_word, token_length=caption_length)
        # conditioning augmentation
        ca_mean, ca_std_log = self.__conditioning_augmentation(sentence_embedding=sentence_embedding)
        __rand = tf.random_normal((__batch, self.__ca_n), mean=0, stddev=1, dtype=tf.float32)
        ca_sample = tf.add(ca_mean, tf.multiply(tf.sqrt(tf.exp(ca_std_log)), __rand))

        # generator
        seed_value = tf.random_normal((__batch, self.__seed_n), mean=0, stddev=1, dtype=tf.float32)
        self.__generator(
            seed_value=seed_value,
            word_embedding=word_embedding,
            sentence_embedding=sentence_embedding
        )

        with tf.name_scope('loss'):
            # loss for conditioning augmentation: KL with unit Gaussian
            loss_kl = - 0.5 * tf.reduce_sum(1 + ca_std_log - tf.square(ca_mean) - tf.exp(ca_std_log), 1)

    def __conditioning_augmentation(self,
                                    sentence_embedding,
                                    scope=None,
                                    reuse=None):
        """ Conditioning augmentation

        :param sentence_embedding: batch, n_hidden * 2
        :param scope:
        :param reuse:
        :return:
        """
        with tf.variable_scope(scope or 'conditioning_augmentation', reuse=reuse):
            mean = util_tf.full_connected(
                sentence_embedding,
                weight_shape=[self.__n_hidden_text_enc * 2, self.__ca_n],
                scope='mean',
                reuse=reuse)

            std = util_tf.full_connected(
                sentence_embedding,
                weight_shape=[self.__n_hidden_text_enc * 2, self.__ca_n],
                scope='std',
                reuse=reuse)
        return mean, std


    def __text_encoder(self,
                       token,
                       token_length,
                       scope=None,
                       reuse=None):
        """ Text encoder (biLSTM)

        :param token: batch, time, feature
        :param token_length: batch,
        :param scope:
        :param reuse:
        :return:
            word_embedding: batch, max_token_length, 2 * n_hidden
            sentence_embedding: batch, 2 * n_hidden
        """
        with tf.variable_scope(scope or 'text_encoder_biLSTM', reuse=reuse):
            feature = tf.transpose(token, perm=[1, 0, 2])  # FusedLSTMCell require (time, batch, feature)
            with tf.variable_scope('fw'):
                lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(self.__n_hidden_text_enc)
                output_fw, state_fw = lstm_cell_fw(feature, dtype=tf.float32, sequence_length=token_length)
            with tf.variable_scope('bw'):
                lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(self.__n_hidden_text_enc)
                lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
                output_bw, state_bw = lstm_cell_bw(feature, dtype=tf.float32, sequence_length=token_length)

        word__embedding = tf.concat([output_fw, output_bw], axis=2)
        word__embedding = tf.transpose(word__embedding, perm=[1, 0, 2])  # (batch, token_len, n_hidden * 2)

        sentence_embedding = tf.concat([state_fw, state_bw], axis=1)  # (batch, n_hidden * 2)
        return word__embedding, sentence_embedding

    def __damsm(self,
                word_embedding,
                images,
                scope=None,
                reuse=None):
        """ Deep Attentional Multi-modal Similarity Model

        :param word_embedding: batch, time, n_hidden * 2
        :param images: batch, scale, scale, 3
        :return:
        """

        with tf.variable_scope(scope or 'damsm', reuse=reuse):


    def __generator(self,
                    seed_value,
                    sentence_embedding,
                    word_embedding,
                    is_training,
                    scope=None,
                    reuse=None):
        """ Attentional Generative Model

        :param seed_value: batch, seed_n
        :param sentence_embedding: batch, n_hidden * 2
        :param word_embedding: batch, time, n_hidden * 2
        :return:
        """

        def unit_block(input_tensor, channel, up_sample_size):
            _x = tf.image.resize_nearest_neighbor(input_tensor, size=[up_sample_size, up_sample_size])
            _x = util_tf.convolution(_x, stride=1, weight_shape=[3, 3, channel, channel], bias=False, padding='SAME')
            _x = util_tf.bn(_x, is_training=is_training)
            _x = util_tf.glu(_x)
            return _x

        ch = 128 * 16
        with tf.variable_scope(scope or 'attentional_generator', reuse=reuse):
            feature = tf.concat([seed_value, sentence_embedding], axis=1)  # batch, n_hidden * 2 + seed_n

            with tf.variable_scope('initial_generator'):
                with tf.variable_scope('shaping_image'):
                    layer = util_tf.full_connected(
                        feature,
                        weight_shape=[self.__seed_n + self.__n_hidden_text_enc * 2, ch * 4 * 4 * 2])
                    layer = util_tf.bn(layer, is_training=is_training)
                    layer = util_tf.glu(layer)  # note that glu reduce channel size to be the half of given tensor
                    layer = tf.reshape(layer, [-1, 4, 4, ch])

                with tf.variable_scope('generate_64'):
                    layer = unit_block(layer, ch, 8)
                    layer = unit_block(layer, ch / 2, 16)
                    layer = unit_block(layer, ch / 4, 34)
                    layer = unit_block(layer, ch / 8, 64)  # batch, 64, 64, ch / 16


    def attention_layer(self,
                        input_tensor,
                        input_width,
                        input_channel,
                        context_tensor,
                        scope=None,
                        reuse=None
                        ):
        """"""
        with tf.variable_scope(scope or 'attention_layer', reuse=reuse):
            context = tf.expand_dims(context_tensor, -1)
            # batch, time, n_hidden, 1 -> # batch, time, 1, input_width
            context = util_tf.convolution(context_tensor,
                                          stride=[1, 1],
                                          weight_shape=[1, self.__n_hidden_text_enc * 2, 1, input_width],
                                          bias=False,
                                          padding='VALID',
                                          scope='context_conv')
            # batch, time, input_width, 1
            context = tf.reshape(context, [-1, self.__length_token, input_width, 1])








