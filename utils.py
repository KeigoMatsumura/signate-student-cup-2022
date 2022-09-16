import os

from sklearn.model_selection import (
    StratifiedKFold,
    KFold,
    GroupKFold,
    StratifiedGroupKFold,
)
import numpy as np
import pandas as pd
import random

import torch

import re
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()


# =====================
# Utils
# =====================
# Seed
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# KFold
def get_stratifiedkfold(train, target_col, n_splits, seed):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    generator = kf.split(train, train[target_col])
    fold_series = []
    for fold, (idx_train, idx_valid) in enumerate(generator):
        fold_series.append(pd.Series(fold, index=idx_valid))
    fold_series = pd.concat(fold_series).sort_index()
    return fold_series

# collatte
def collatte(inputs, labels=None):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    if not labels is None:
        inputs = {
            "input_ids" : inputs['input_ids'][:,:mask_len],
            "attention_mask" : inputs['attention_mask'][:,:mask_len],
        }
        labels =  labels[:,:mask_len]
        return inputs, labels, mask_len
                
    else:
        inputs = {
            "input_ids" : inputs['input_ids'][:,:mask_len],
            "attention_mask" : inputs['attention_mask'][:,:mask_len],
        }
        return inputs, mask_len

# cleaning
def cleaning(texts):
    clean_texts = []
    for text in texts:
        # htmlタグを削除
        text = remove_tag(text)
        #アルファベット以外をスペースに置き換え
        clean_punc = re.sub(r'[^a-zA-Z]', ' ', text)
        #単語長が3文字以下のものは削除する
        clean_short_tokenized = [word for word in clean_punc.split() if len(word) > 3]
        #ステミング
        #clean_normalize = [stemmer.stem(word) for word in clean_short_tokenized]
        #単語同士をスペースでつなぎ, 文章に戻す
        clean_text = ' '.join(clean_short_tokenized)
        clean_texts.append(clean_text)
    return clean_texts

def remove_tag(x):
    p = re.compile(r"<[^>]*?>")
    return p.sub('',x)
