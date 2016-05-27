#!/usr/bin/env python
# -*- coding: utf-8 -*-

import common as c
from common import Logger as Log
import eisner
import numpy as np


def train(sentences, iteration=10):
    edge_feats_list, featdict = c.extract(sentences)
    weights = np.zeros(len(featdict))

    size = len(sentences)

    for i in range(1, iteration + 1):
        Log.i("Training iteration: %d of %d" % (i, iteration))
        count = 0
        for sentence, edge_feats in zip(sentences, edge_feats_list):
            count = count + 1
            Log.i("\tTraining sentence: %d of %d" % (count, size))
            predicts = eisner.parse(sentence, weights, edge_feats)
            weights = update(weights, predicts, sentence, edge_feats)
            """
            print(predicts)
            for feat, index in featdict.items():
                print(index, feat, weights[index])
            """
    return featdict, weights


def update(weights, predicts, sentence, edge_feats):
    for i, word in enumerate(sentence):
        h_predict = predicts[i]
        h_gold = word.head
        if h_predict != h_gold:
            for fidx in edge_feats.get((h_predict, i), []):
                weights[fidx] = weights[fidx] - 1
            for fidx in edge_feats.get((h_gold, i), []):
                weights[fidx] = weights[fidx] + 1
    return weights


def main(args):
    if len(args) < 2:
        Log.w("Arguments Error")
        return
    path = args[0]
    iteration = int(args[1])
    Log.i("Training file: %s" % path)
    Log.i("----------------")
    sentences = c.readconllfile(path)
    featdict, weights = train(sentences, iteration)
    c.save(featdict, weights)


if __name__ == "__main__":
    Log.setConfig(loglevel=Log.DEBUG, verbose=True)
    main(c.readargs())
    Log.finalize()
