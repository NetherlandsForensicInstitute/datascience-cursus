#!/usr/bin/env python3
import lir
from lir import FeatureData, Transformer
from lir.aggregation import plot_lr_histogram
from lir.algorithms.kde import KDECalibrator
from lir.algorithms.logistic_regression import LogitCalibrator
from lir.lrsystems import LRSystem, BinaryLRSystem
from lir.plotting import lr_histogram
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(seed=0)

value_of_e = .5

DATA_SIZE = 200

ss_data = rng.normal(loc=1.5, scale=1, size=DATA_SIZE)
ss_data = FeatureData(features=ss_data, labels=np.ones(DATA_SIZE, dtype=np.int8))

ds_data = rng.normal(loc=-2, scale=1.5, size=DATA_SIZE)
ds_data[ds_data < -5] = -5
ds_data = FeatureData(features=ds_data, labels=np.zeros(DATA_SIZE, dtype=np.int8))


def plot_curve(ax, xvalues, yvalues, value_of_e: float, likelihood_of_e: float, color: str):
  assert isinstance(value_of_e, float)
  assert isinstance(likelihood_of_e, float), f"type is {type(likelihood_of_e)}"
  ax.plot(xvalues, yvalues, color=color)
  ax.plot(value_of_e, likelihood_of_e, 'go')
  ytlocs, ytlabels = plt.yticks()
  ytlocs = list(ytlocs) + [likelihood_of_e]
  ytlabels = list(ytlabels) + [f'{likelihood_of_e:.2f}']
  ax.set_yticks(ytlocs, ytlabels)
  ax.vlines(value_of_e, plt.ylim()[0], likelihood_of_e, linestyle='--', color='g')
  ax.hlines(likelihood_of_e, plt.xlim()[0], value_of_e, linestyle='--', color='g')


def show_hist(
    filename: str,
    value_of_e: float,
    background_data: FeatureData,
    density: bool = False,
    kde: KernelDensity = None,
    show_logit: bool = False,
    select_class: int | None = None,
):
  fig, ax = plt.subplots()
  colors = ['r', 'b']
  if kde or show_logit:
    density = True

  for i in np.unique(background_data.labels):
    if select_class is None or select_class == i:
      data = background_data[background_data.labels==i].features
      ax.hist(data, bins=20, alpha=.4, edgecolor='black', density=density, color=colors[i])

  if kde:
    for i in np.unique(background_data.labels):
      if select_class is None or select_class == i:
        data = background_data[background_data.labels==i]
        kde.fit(data.features)
        likelihood_of_e = np.exp(kde.score_samples([[value_of_e]]))[0]
        xvalues = np.linspace(np.min(background_data.features), np.max(background_data.features), 100)
        yvalues = kde.score_samples(xvalues.reshape(-1, 1))
        plot_curve(ax, xvalues, np.exp(yvalues), value_of_e, likelihood_of_e, colors[i])
  elif show_logit:
    logit = LogisticRegression(class_weight='balanced')
    logit.fit(background_data.features, background_data.labels)

    xvalues = np.linspace(np.min(background_data.features), np.max(background_data.features), 100)
    pvalues = logit.predict_proba(xvalues.reshape(-1, 1))
    pvalue_of_e = logit.predict_proba([[value_of_e]])

    for i in np.unique(background_data.labels):
      if select_class is None or select_class == i:
        plot_curve(ax, xvalues, pvalues[:, i], value_of_e, pvalue_of_e[0, i], colors[i])
  else:
    plt.vlines(value_of_e, *plt.ylim(), linestyle='--', color='g')

  locs, labels = plt.xticks()
  ax.set_xticks(list(locs) + [value_of_e], list(labels) + [f'E={value_of_e}'])
  ax.set_xlim(np.min(background_data.features)-.2, np.max(background_data.features)+.2)
  ax.set_xlabel('score')
  ax.set_ylabel('relative frequency' if density else 'count')

  plt.savefig(filename)
  #plt.show()
  plt.close(fig)


def plot_score2lr(filename: str, data: FeatureData, tr: Transformer, figsize = None):
  fig, ax = plt.subplots(figsize=figsize)
  llrs = tr.fit_apply(data)
  xvalues = np.linspace(np.min(data.features), np.max(data.features), 100)
  yvalues = tr.apply(FeatureData(features=xvalues))
  ax.plot(xvalues, yvalues.features, color='g')
  #ax.plot(data.features, llrs.features, color='g', marker='o')
  ax.grid()
  ax.set_xlabel('score')
  ax.set_ylabel('log(LR)')
  ax.get_tightbbox()
  ax.set_ylim(-4, 4)
  plt.savefig(filename)
  #plt.show()
  plt.close(fig)


kde2 = KernelDensity(kernel='gaussian', bandwidth=.2)
kde5 = KernelDensity(kernel='gaussian', bandwidth=.5)

show_hist('00_hist.png', value_of_e, ds_data + ss_data)
show_hist('01_ss_hist.png', value_of_e, ds_data + ss_data, select_class=1)

show_hist('10_ss_kde.png', value_of_e, ds_data + ss_data, select_class=1, kde=kde5)
show_hist('11_kde.png', value_of_e, ds_data + ss_data, kde=kde5)
show_hist('12_kde_small_bandwidth.png', value_of_e, ds_data + ss_data, kde=kde2)
show_hist('13_logit.png', value_of_e, ds_data + ss_data, show_logit=True)

plot_score2lr('20_logit_score2lr.png', ds_data + ss_data, LogitCalibrator(), figsize=(8, 2))
plot_score2lr('21_kde5_score2lr.png', ds_data + ss_data, KDECalibrator(bandwidth=.5), figsize=(8, 2))
plot_score2lr('22_kde2_score2lr.png', ds_data + ss_data, KDECalibrator(bandwidth=.2), figsize=(8, 2))

with lir.plotting.savefig('30_lr_histogram') as canvas:
  canvas.lr_histogram(LogitCalibrator().fit_apply(ds_data + ss_data))