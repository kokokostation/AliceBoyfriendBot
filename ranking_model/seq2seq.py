import tensorflow as tf
import tensorflow_addons as tfa

from embedders.flavor_embedder import flavor_embedder
from model.interfaces.model import FusedModel, RankingModelBoxIntruder, RankingModel


@flavor_embedder
def encoder(embeddings, sent_lens, mp):
    encoder_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(mp['hid_size'])

    encoder_outputs, encoder_state = tf.compat.v1.nn.dynamic_rnn(encoder_cell, embeddings,
                                                                 sent_lens, dtype=tf.float32)

    return encoder_outputs, encoder_state


@flavor_embedder
def decoder(embeddings, sent_lens, reply_params,
            encoder_outputs, encoder_state, encoder_sent_lens,
            context_params):
    decoder_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(reply_params['hid_size'])
    helper = tfa.seq2seq.TrainingHelper(embeddings, sent_lens)
    initial_state = encoder_state

    if reply_params.get('attention') == 'bahdanau':
        encoder_hid_size = context_params['hid_size']

        attention_mechanism = tfa.seq2seq.BahdanauAttention(
            encoder_hid_size, encoder_outputs,
            memory_sequence_length=encoder_sent_lens)

        decoder_cell = tfa.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism,
            attention_layer_size=encoder_hid_size)

        initial_state = decoder_cell.zero_state(tf.shape(input=embeddings)[0], tf.float32) \
            .clone(cell_state=encoder_state)

    decoder = tfa.seq2seq.BasicDecoder(decoder_cell, helper, initial_state)
    _, decoder_state, _ = tfa.seq2seq.dynamic_decode(decoder)

    if reply_params.get('attention') == 'bahdanau':
        decoder_state = decoder_state.cell_state

    return decoder_state.c


class Seq2seqRankingModel(RankingModel, FusedModel, RankingModelBoxIntruder):
    def make_context(self, context, context_params):
        return encoder(context[0], 'ranking_encoder', context_params) + (context[0][-1],)

    def make_reply(self, context_outputs, model_params, reply):
        encoder_outputs, encoder_state, encoder_sent_lens = context_outputs

        return decoder(reply, 'ranking_decoder', model_params['reply'],
                       encoder_outputs, encoder_state, encoder_sent_lens, model_params['context'])
