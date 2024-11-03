# -*- coding: utf-8 -*-
# Time  : 2022/4/11
# Author: slmu
# Email : mushanlei.msl@alibaba-inc.com

import torch
import numpy as np
from sklearn import metrics


class BaseEvaluator(object):
    def __init__(self):
        super(BaseEvaluator, self).__init__()
        self.scores = []
        self.labels = []

    def accumulate(self, score, label):
        self.scores.append(score)
        self.labels.append(label)

    def calculate_metrics(self):
        results = dict()
        scores = torch.cat(self.scores).cpu().numpy()
        labels = torch.cat(self.labels).cpu().numpy().astype(np.int32)
        self.scores = []
        self.labels = []
        auc = metrics.roc_auc_score(labels, scores)
        results['auc'] = round(auc, 4)
        return results
