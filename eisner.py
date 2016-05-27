#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys


def parse(sentence, weights, edge_feats):
    n = len(sentence)
    scores = {}
    INF = sys.maxsize

    for i_head in range(0, n):
        for i_mod in range(0, n):
            if i_head != i_mod:
                score = 0
                for fidx in edge_feats[(i_head, i_mod)]:
                    score = score + weights[fidx]
            else:
                score = -INF
            scores[(i_head, i_mod)] = score

    comp_rh = np.empty((n, n), dtype=np.int)  # C[i][j][←][0]
    comp_lh = np.empty((n, n), dtype=np.int)  # C[i][j][→][0]
    incomp_rh = np.empty((n, n), dtype=np.int)  # C[i][j][←][1]
    incomp_lh = np.empty((n, n), dtype=np.int)  # C[i][j][→][1]

    bp_comp_rh = np.empty((n, n), dtype=np.int)
    bp_comp_lh = np.empty((n, n), dtype=np.int)
    bp_incomp_rh = np.empty((n, n), dtype=np.int)
    bp_incomp_lh = np.empty((n, n), dtype=np.int)

    for i in range(0, n):
        comp_rh[i][i] = 0
        comp_lh[i][i] = 0
        incomp_rh[i][i] = 0
        incomp_lh[i][i] = 0

    for m in range(1, n):
        for i in range(0, n):
            j = i + m
            if j >= n:
                break

            # C[i][j][←][1] : right head, incomplete
            max = -INF
            bp = -1
            ex = scores[(j, i)]
            for k in range(i, j):
                score = comp_lh[i][k] + comp_rh[k + 1][j] + ex
                if score > max:
                    max = score
                    bp = k
            incomp_rh[i][j] = max
            bp_incomp_rh[i][j] = bp

            # C[i][j][→][1] : left head, incomplete
            max = -INF
            bp = -1
            ex = scores[(i, j)]
            for k in range(i, j):
                score = comp_lh[i][k] + comp_rh[k + 1][j] + ex
                if score > max:
                    max = score
                    bp = k
            incomp_lh[i][j] = max
            bp_incomp_lh[i][j] = bp

            # C[i][j][←][0] : right head, complete
            max = -INF
            bp = -1
            for k in range(i, j):
                score = comp_rh[i][k] + incomp_rh[k][j]
                if score > max:
                    max = score
                    bp = k
            comp_rh[i][j] = max
            bp_comp_rh[i][j] = bp

            # C[i][j][→][0] : left head, complete
            max = -INF
            bp = -1
            for k in range(i + 1, j + 1):
                score = incomp_lh[i][k] + comp_lh[k][j]
                if score > max:
                    max = score
                    bp = k
            comp_lh[i][j] = max
            bp_comp_lh[i][j] = bp

    heads = [None] * n
    heads[0] = -1

    def _backtrack(i, j, lh, c):
        """
            lh: right head = 0, left head = 1
            c: complete = 0, incomplete = 1
        """
        if i == j:
            return
        elif lh == 1 and c == 0:  # comp_lh
            k = bp_comp_lh[i][j]
            heads[k] = i
            heads[j] = k
            _backtrack(i, k, 1, 1)
            _backtrack(k, j, 1, 0)
        if lh == 0 and c == 0:  # comp_rh
            k = bp_comp_rh[i][j]
            heads[k] = j
            heads[i] = k
            _backtrack(i, k, 0, 0)
            _backtrack(k, j, 0, 1)
        elif lh == 1 and c == 1:  # incomp_lh
            k = bp_incomp_lh[i][j]
            heads[j] = i
            _backtrack(i, k, 1, 0)
            _backtrack(k + 1, j, 0, 0)
        elif lh == 0 and c == 1:  # incomp_rh
            k = bp_incomp_rh[i][j]
            heads[i] = j
            _backtrack(i, k, 1, 0)
            _backtrack(k + 1, j, 0, 0)

    _backtrack(0, (n - 1), 1, 0)
    return heads
