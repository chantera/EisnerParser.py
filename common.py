#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import logging
import pickle
import sys
import uuid


class Model:

    def __init__(self, featdict, weights):
        self.featdict = featdict
        self.weights = weights


class Word:

    def __init__(self, **args):
        self.id = int(args['id'])
        self.form = args['form']
        self.lemma = args['lemma']
        self.cpostag = args['cpostag']
        self.postag = args['postag']
        self.feats = args['feats']
        self.head = int(args['head'])
        self.deprel = args['deprel']

    @staticmethod
    def createRoot():
        root = Word(
            id=0,
            form="<ROOT>",
            lemma="<ROOT>",
            cpostag="ROOT",
            postag="ROOT",
            feats="",
            head=-1,
            deprel="ROOT"
        )
        return root


class _FeatureRegistry:
    __instance = None

    def _initialize(self):
        self._featdict = {}
        self._n_feats = -1

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            instance = object.__new__(cls)
            instance._initialize()
            cls.__instance = instance
        return cls.__instance

    def _getFeatures(self, head, mod):
        h_word = "h_word=" + head.form
        h_pos = "h_pos=" + head.postag
        m_word = "m_word=" + mod.form
        m_pos = "m_pos=" + mod.postag

        template = [
            # unigram
            ":".join([h_word, h_pos]),
            ":".join([h_word]),
            ":".join([h_pos]),
            ":".join([m_word, m_pos]),
            ":".join([m_word]),
            ":".join([m_pos]),
            # bigram
            ":".join([h_word, h_pos, m_word, m_pos]),
            ":".join([h_pos, m_word, m_pos]),
            ":".join([h_word, m_word, m_pos]),
            ":".join([h_word, h_pos, m_pos]),
            ":".join([h_word, h_pos, m_word]),
            ":".join([h_word, m_word]),
            ":".join([h_pos, m_pos])
        ]
        return template

    def _getFIndexes(self, features):
        indexes = []
        for feature in features:
            if feature in self._featdict:
                indexes.append(self._featdict[feature])
            else:
                self._n_feats = self._n_feats + 1
                index = self._n_feats
                self._featdict[feature] = index
                indexes.append(index)
        return indexes

    def extract(self, sentences):
        edge_feats_list = []
        for sentence in sentences:
            edge_feats = {}
            n = len(sentence)
            for i_head in range(0, n):
                for i_mod in range(0, n):
                    if i_head != i_mod:
                        edge_feats[(i_head, i_mod)] = self._getFIndexes(self._getFeatures(sentence[i_head], sentence[i_mod]))
            edge_feats_list.append(edge_feats)
        return edge_feats_list, self._featdict


def extract(sentences):
    return _FeatureRegistry.getInstance().extract(sentences)


def readconllfile(path):
    sentences = []
    with open(path, 'r') as f:
        words = [Word.createRoot()]
        for line in f:
            if not line.strip():
                sentences.append(words)
                words = [Word.createRoot()]
            else:
                cols = line.split("\t")
                word = Word(
                    id=cols[0],
                    form=cols[1],
                    lemma=cols[2],
                    cpostag=cols[3],
                    postag=cols[4],
                    feats=cols[5],
                    head=cols[6],
                    deprel=cols[7]
                )
                words.append(word)
    return sentences


def save(featdict, weights):
    filename = datetime.now().strftime("%Y%m%d%H%M%S.model.txt")
    model = Model(featdict, weights)
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load(path):
    model = None
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model.featdict, model.weights


def gen_hexid():
    return uuid.uuid4().hex[:6]


class Logger:
    __instance = None

    FATAL = logging.FATAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG

    LABELS = {
        ERROR: 'error',
        WARNING: 'warn',
        INFO: 'info',
        DEBUG: 'debug',
    }

    # global config
    _verbose = False
    _loglevel = ERROR

    def __init__(self):
        raise NotImplementedError()

    @classmethod
    def finalize(cls):
        cls._getInstance()._stop()
        Logger.__instance = None

    def _initialize(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        format = "%(asctime)-15s\t%(accessid)s\t[%(priority)s]\t%(message)s"
        logfile = "logs/" + datetime.now().strftime("%Y%m%d") + ".log"
        logging.basicConfig(format=format, level=Logger._loglevel, filename=logfile)
        self._logger = logging.getLogger(__name__)
        if Logger._verbose:
            self._logger.addHandler(logging.StreamHandler())
        self._start()
        self._initialized = True

    @classmethod
    def _getInstance(cls):
        if cls.__instance is None:
            instance = object.__new__(cls)
            instance._initialize()
            cls.__instance = instance
        return cls.__instance

    @classmethod
    def setConfig(cls, loglevel=_loglevel, verbose=_verbose):
        cls._loglevel = loglevel
        cls._verbose = verbose

    def _start(self):
        now = datetime.now()
        self._accessid = gen_hexid()
        self._uniqueid = "UNIQID"
        self._accesssec = now
        self._accesstime = now.strftime("%Y/%m/%d %H:%M:%S.%f %Z")
        message = "LOG Start with ACCESSID=[%s] UNIQUEID=[%s] ACCESSTIME=[%s]"
        self._log(Logger.INFO, message % (self._accessid, self._uniqueid, now))

    def _stop(self):
        processtime = '%3.9f' % (datetime.now() - self._accesssec).total_seconds()
        message = "LOG End with ACCESSID=[%s] UNIQUEID=[%s] ACCESSTIME=[%s] PROCESSTIME=[%s]\n"
        self._log(Logger.INFO, message % (self._accessid, self._uniqueid, self._accesstime, processtime))

    def _log(self, level, message):
        extras = {
            'accessid': self._accessid,
            'priority': Logger.LABELS[level]
        }
        self._logger.log(level, message, extra=extras)

    @classmethod
    def i(cls, message):
        cls._getInstance()._log(Logger.INFO, message)

    @classmethod
    def w(cls, message):
        cls._getInstance()._log(Logger.WARNING, message)

    @classmethod
    def d(cls, message):
        cls._getInstance()._log(Logger.DEBUG, message)

    @classmethod
    def v(cls, message):
        cls.d(message)


def readargs():
    return sys.argv[1:]
