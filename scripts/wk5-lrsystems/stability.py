#!/usr/bin/env python3

from lir import FeatureData
from lir.data_strategies import TrainTestSplit
from lir.lrsystems import BinaryLRSystem
from lir.algorithms.logistic_regression import LogitCalibrator
from lir.plotting import lr_histogram
from lir import metrics
import numpy as np
from matplotlib import pyplot as plt
from functools import cached_property


class Factory:
    def __init__(self, seed: int):
        self.seed = seed

    @cached_property
    def data(self) -> FeatureData:
        rng = np.random.default_rng(seed=self.seed)
        data = rng.normal(loc=0, scale=1, size=120)
        labels = np.zeros(len(data))
        data[len(data) // 2:] += 2
        labels[len(data) // 2:] = 1
        return FeatureData(features=data, labels=labels)

    @property
    def lrsystem(self):
        return BinaryLRSystem(pipeline=LogitCalibrator(random_state=self.seed))

    @cached_property
    def test_llrs(self):
        splitter = TrainTestSplit(test_size=.5, seed=self.seed)
        for training_data, test_data in splitter.apply(self.data):
            return self.lrsystem.fit(training_data).apply(test_data)

    @cached_property
    def cllr(self):
        return metrics.cllr(self.test_llrs)


factories = [Factory(i) for i in range(10)]
worst = max(factories, key=lambda f: f.cllr)
best = min(factories, key=lambda f: f.cllr)
factories = iter([best, worst, factories[0], factories[1]])

seed = 0
fig, axs = plt.subplots(2, 2)
split = TrainTestSplit(test_size=.5)
for row in axs:
    for ax in row:
        factory = next(factories)
        lr_histogram(ax, factory.test_llrs, bins=10)
        ax.set_title(f'seed={factory.seed}; cllr={factory.cllr:.2f}')
fig.tight_layout()
plt.show()