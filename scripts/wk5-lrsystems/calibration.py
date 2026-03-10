#!/usr/bin/env python3

from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(seed=0)

value_of_e = .5

ss_data = np.zeros(200)
ss_data += rng.normal(loc=1.5, scale=1, size=ss_data.shape)

ds_data = np.zeros(200)
ds_data += rng.normal(loc=-2, scale=1.5, size=ss_data.shape)
ds_data[ds_data < -5] = -5


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
    value_of_e: float,
    background_data: list[np.ndarray],
    density: bool = False,
    show_kde: bool = False,
    kde_bandwidth: float = .5,
    show_logit: bool = False,
    select_class: int | None = None,
):
  fig, ax = plt.subplots()
  colors = ['r', 'b']
  if show_kde or show_logit:
    density = True

  for i, data in enumerate(background_data):
    if select_class is None or select_class == i:
      ax.hist(data, bins=20, alpha=.4, edgecolor='black', density=density, color=colors[i])

  if show_kde:
    for i, data in enumerate(background_data):
      if select_class is None or select_class == i:
        kernel = KernelDensity(kernel='gaussian', bandwidth=kde_bandwidth).fit(data.reshape(-1, 1))
        likelihood_of_e = np.exp(kernel.score_samples([[value_of_e]]))[0]
        xvalues = np.linspace(np.min(background_data), np.max(background_data), 100)
        yvalues = kernel.score_samples(xvalues.reshape(-1, 1))
        plot_curve(ax, xvalues, np.exp(yvalues), value_of_e, likelihood_of_e, colors[i])
  elif show_logit:
    logit = LogisticRegression(class_weight='balanced')
    logit.fit(np.concatenate(background_data).reshape(-1, 1), np.concatenate(list(np.ones(data.shape)*i for i, data in enumerate(background_data))))

    xvalues = np.linspace(np.min(background_data), np.max(background_data), 100)
    pvalues = logit.predict_proba(xvalues.reshape(-1, 1))
    pvalue_of_e = logit.predict_proba([[value_of_e]])

    for i, data in enumerate(background_data):
      if select_class is None or select_class == i:
        plot_curve(ax, xvalues, pvalues[:, i], value_of_e, pvalue_of_e[0, i], colors[i])
  else:
    plt.vlines(value_of_e, *plt.ylim(), linestyle='--', color='g')

  locs, labels = plt.xticks()
  ax.set_xticks(list(locs) + [value_of_e], list(labels) + [f'E={value_of_e}'])
  ax.set_xlim(np.min(background_data)-.2, np.max(background_data)+.2)
  ax.set_xlabel('score')
  ax.set_ylabel('relative frequency' if density else 'count')

  plt.show()
  plt.close(fig)


show_hist(value_of_e, [ds_data, ss_data], select_class=1)
show_hist(value_of_e, [ds_data, ss_data], select_class=1, show_kde=True)
show_hist(value_of_e, [ds_data, ss_data], show_kde=True)
show_hist(value_of_e, [ds_data, ss_data], show_kde=True, kde_bandwidth=.2)
show_hist(value_of_e, [ds_data, ss_data], show_kde=True, kde_bandwidth=.2)
show_hist(value_of_e, [ds_data, ss_data], show_logit=True)
