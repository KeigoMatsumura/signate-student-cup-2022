import os
import gc
import re
import sys
import json
import time
import shutil
import joblib
import random
import warnings
warnings.filterwarnings('ignore')
from ast import literal_eval

import numpy as np
import pandas as pd

import scipy
import itertools
from pathlib import Path
from glob import glob
from tqdm.auto import tqdm
from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    GroupKFold,
    StratifiedGroupKFold,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

from datasets import BERTDataset
from datasets import BERTModel
from utils import collatte

def inferring(cfg, test):
    # 損失関数
    criterion = nn.CrossEntropyLoss(weight=cfg.class_weights)

    print('\n'.join(cfg.model_weights))
    sub_pred = np.zeros((len(test), 4), dtype=np.float32)
    for fold, model_weight in enumerate(cfg.model_weights):
        # dataset, dataloader
        test_dataset = BERTDataset(
            cfg,
            test['description'].to_numpy()
        )
        test_loader = DataLoader(
            dataset=test_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=False,
            pin_memory=True
        )
        model = BERTModel(cfg, criterion)
        model.load_state_dict(torch.load(model_weight))
        model = model.to(cfg.device)

        model.eval()
        fold_pred = []
        with torch.no_grad():
            for inputs in tqdm(test_loader, total=len(test_loader)):
                inputs, max_len = collatte(inputs)
                for k, v in inputs.items():
                    inputs[k] = v.to(cfg.device)
                with autocast():
                    output = model(inputs)
                output = output.softmax(axis=1).detach().cpu().numpy()
                fold_pred.append(output)
        fold_pred = np.concatenate(fold_pred)
        np.save(os.path.join(cfg.EXP_PREDS, f'sub_pred_fold{fold}.npy'), fold_pred)
        sub_pred += fold_pred / len(cfg.model_weights)
        del model; gc.collect()
    np.save(os.path.join(cfg.EXP_PREDS, f'sub_pred.npy'), sub_pred)
    return sub_pred
