#!/usr/bin/env python
# -*- coding: utf-8 -*-

import common as c
from common import Logger as Log
import eisner


def parse(sentences, weights):
    results = []
    edge_feats_list, featdict = c.extract(sentences)

    size = len(sentences)
    count = 0
    correct = 0
    total = 0

    for sentence, edge_feats in zip(sentences, edge_feats_list):
        count = count + 1
        Log.i("Parsing sentence: %d of %d" % (count, size))
        result = []
        heads = eisner.parse(sentence, weights, edge_feats)
        for index, word in enumerate(sentence):
            if index == 0:
                continue
            token = [
                str(word.id),
                word.form,
                word.lemma,
                word.cpostag,
                word.postag,
                word.feats,
                str(heads[index]),
                word.deprel
            ]
            line = "\t".join(token)
            if heads[index] == word.head:
                correct = correct + 1
            total = total + 1
            result.append(line)
        results.append(result)
    return results, float(correct / total)


def main(args):
    if len(args) < 2:
        Log.w("Arguments Error")
        return
    path = args[0]
    mfile = args[1]
    Log.i("Training file: %s" % path)
    Log.i("----------------")
    sentences = c.readconllfile(path)
    Log.i("Model file: %s" % mfile)
    Log.i("----------------")
    featdict, weights = c.load(mfile)
    output, score = parse(sentences, weights)
    for each in output:
        for token in each:
            print(token)
        print()
    Log.i("[DONE] accuracy: {:.2%}".format(score))


if __name__ == "__main__":
    Log.setConfig(loglevel=Log.DEBUG, verbose=True)
    main(c.readargs())
    Log.finalize()
