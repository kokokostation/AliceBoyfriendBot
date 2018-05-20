import numpy as np
from tqdm import tqdm


def make_replies(model, reply_gen):
    replies = []
    vectors = []
    priors = []
    index = set()

    for full, [batch] in tqdm(reply_gen):
        vs = model.reply_embedding(batch)

        for (msg, prior), v in zip(full, vs):
            if msg not in index:
                priors.append(prior)
                replies.append(msg)
                vectors.append(v)
                index.add(msg)

    vectors = {
        'vectors': np.array(vectors),
        'priors': np.array(priors)
    }

    data = {
        'replies': replies,
        'vectors': vectors
    }

    return data


def k_best(k, scores):
    inds = np.argpartition(scores, -k)[-k:]

    return list(inds[np.argsort(-scores[inds])])