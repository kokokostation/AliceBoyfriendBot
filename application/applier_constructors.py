from functools import partial
import tensorflow as tf

from application.applier import RankingApplier, Applier
from model.lstm_embedder import LSTMModel
from model.baseline import Baseline
from application.ranking_model import RankingApplicationModel
from application.model import ApplicationModel
from train_utils.container import ModelContainer
from batch_generator.flavors import RECURRENT_FUSED_PRIOR_REPLY
from ranking_model.baseline import RankingBaseline
from batch_generator.flavors import RANKING_PLAIN_TRAIN, RANKING_FUSED_TRAIN, PLAIN_PRIOR_REPLY
from application.context_preparer import FusedContextPreparer, ContextPreparer
from dataset.reddit import convert
from application.prior_storage import PriorStorage, RestrictingPriorStorage
from application.interfaces.vector_storage import ReplyContainer, RestrictingContainer
from application.utils import k_best


def make_baseline(config_dir='/data/reddit/models/x_prod/',
                  ckpt='reddit_tf_weights_baseline_check'):
    with tf.device("/device:GPU:0"):
        plain_mc = ModelContainer(config_dir, ckpt=ckpt, dill=True)
        am = ApplicationModel(plain_mc, Baseline(plain_mc, False), PLAIN_PRIOR_REPLY(plain_mc))

        vector_storage = RestrictingPriorStorage(0.1, 5)
        plain_context_preparer = ContextPreparer(convert)

        ap = Applier(am, vector_storage, plain_context_preparer, plain_mc)

    return ap


def make_lstm(config_dir='/data/reddit/models/independant_for_ranking/',
              ckpt='reddit_tf_weights'):
    with tf.device("/device:GPU:0"):
        plain_mc = ModelContainer(config_dir, ckpt=ckpt, dill=True)
        am = ApplicationModel(plain_mc, LSTMModel(plain_mc, False),
                              RECURRENT_FUSED_PRIOR_REPLY(plain_mc))

        vector_storage = RestrictingPriorStorage(0.1, 5)
        plain_context_preparer = FusedContextPreparer(convert)

        ap = Applier(am, vector_storage, plain_context_preparer, plain_mc)

    return ap


def make_ranking(lstm_config_dir='/data/reddit/models/independant_for_ranking/',
                 ranking_config_dir='/data/reddit/models/ranking_baseline_final/',
                 ckpt='reddit_tf_weights',
                 hypos=20, verbose=True):
    with tf.device("/device:GPU:0"):
        plain_mc = ModelContainer(lstm_config_dir, ckpt=ckpt, dill=True)
        am = ApplicationModel(plain_mc, LSTMModel(plain_mc, False),
                              RECURRENT_FUSED_PRIOR_REPLY(plain_mc))

        ranking_mc = ModelContainer(ranking_config_dir, ckpt=ckpt, dill=True)
        mp = ranking_mc.model_params()
        mp['multiplier'] = hypos
        ranking_mc.write('model_params', mp)
        ram = RankingApplicationModel(ranking_mc, RankingBaseline(ranking_mc), RANKING_PLAIN_TRAIN)

        plain_context_preparer = FusedContextPreparer(convert)
        ranking_context_preparer = ContextPreparer(convert)
        vector_storage = PriorStorage(0.1,
                                      partial(k_best, ranking_mc.model_params()['multiplier']))
        container = RestrictingContainer(5)

        ap = RankingApplier(am, vector_storage, plain_context_preparer, ranking_context_preparer,
                            plain_mc, ram, container, verbose)

    return ap