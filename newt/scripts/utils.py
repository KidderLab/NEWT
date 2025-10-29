#!/usr/bin/env python3
"""
Minimal utils.parallel.map replacement for NEWT L1000 model
"""
import multiprocessing as mp

class parallel:
    @staticmethod
    def map(func, items, n_CPU=1, progress=False):
        if n_CPU <= 1:
            return [func(x) for x in items]
        with mp.Pool(processes=n_CPU) as pool:
            return pool.map(func, items)
