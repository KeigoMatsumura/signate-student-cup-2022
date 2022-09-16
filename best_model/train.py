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

import config
from config import setup
from utils import get_stratifiedkfold
from utils import set_seed
from utils import collatte
from datasets import BERTDataset
from datasets import BERTModel
from infer import inferring


def training(cfg, train):
    # =====================
    # Training
    # =====================
    set_seed(cfg.seed)
    oof_pred = np.zeros((len(train), 4), dtype=np.float32)
    
    # 損失関数
    criterion = nn.CrossEntropyLoss()

    for fold in cfg.trn_fold:
        # Dataset,Dataloaderの設定
        train_df = train.loc[cfg.folds!=fold]
        valid_df = train.loc[cfg.folds==fold]
        train_idx = list(train_df.index)
        valid_idx = list(valid_df.index)

        train_dataset = BERTDataset(
            cfg,
            train_df['description'].to_numpy(), 
            train_df['jobflag'].to_numpy(),
        )
        valid_dataset = BERTDataset(
            cfg, 
            valid_df['description'].to_numpy(), 
            valid_df['jobflag'].to_numpy()
        )
        train_loader = DataLoader(
            dataset=train_dataset, 
            batch_size=cfg.batch_size, 
            shuffle=True,
            pin_memory=True,
            drop_last=True
        )
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False
        )

        # 初期化
        best_val_preds = None
        best_val_score = -1

        # modelの読み込み
        model = BERTModel(cfg, criterion)
        model = model.to(cfg.device)

        # optimizer，schedulerの設定
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = []
        optimizer_grouped_parameters.append({
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
            'weight_decay': cfg.weight_decay
        })
        optimizer_grouped_parameters.append({
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
            'weight_decay': 0.0
        })
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=cfg.lr,
            betas=cfg.beta,
            weight_decay=cfg.weight_decay,
        )
        num_train_optimization_steps = int(
            len(train_loader) * cfg.n_epochs // cfg.gradient_accumulation_steps
        )
        num_warmup_steps = int(num_train_optimization_steps * cfg.num_warmup_steps_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_optimization_steps
        )
        num_eval_step = len(train_loader) // cfg.num_eval + cfg.num_eval
        
        for epoch in range(cfg.n_epochs):
            # training
            print(f"# ============ start epoch:{epoch} ============== #")
            model.train() 
            val_losses_batch = []
            scaler = GradScaler()
            with tqdm(train_loader, total=len(train_loader)) as pbar:
                for step, (inputs, labels) in enumerate(pbar):
                    inputs, max_len = collatte(inputs)
                    for k, v in inputs.items():
                        inputs[k] = v.to(cfg.device)
                    labels = labels.to(cfg.device)

                    optimizer.zero_grad()
                    with autocast():
                        output, loss = model(inputs, labels)
                    pbar.set_postfix({
                        'loss': loss.item(),
                        'lr': scheduler.get_lr()[0]
                    })

                    if cfg.gradient_accumulation_steps > 1:
                        loss = loss / cfg.gradient_accumulation_steps
                    scaler.scale(loss).backward()
                    if cfg.clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), 
                            cfg.clip_grad_norm
                        )
                    if (step+1) % cfg.gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                
            # evaluating
            val_preds = []
            val_losses = []
            val_nums = []
            model.eval()
            with torch.no_grad():
                with tqdm(valid_loader, total=len(valid_loader)) as pbar:
                    for (inputs, labels) in pbar:
                        inputs, max_len = collatte(inputs)
                        for k, v in inputs.items():
                            inputs[k] = v.to(cfg.device)
                        labels = labels.to(cfg.device)
                        with autocast():
                            output, loss = model(inputs, labels)
                        output = output.sigmoid().detach().cpu().numpy()
                        val_preds.append(output)
                        val_losses.append(loss.item() * len(labels))
                        val_nums.append(len(labels))
                        pbar.set_postfix({
                            'val_loss': loss.item()
                        })

            val_preds = np.concatenate(val_preds)
            val_loss = sum(val_losses) / sum(val_nums)
            score = f1_score(np.argmax(val_preds, axis=1), valid_df['jobflag'], average='macro')
            val_log = {
                'val_loss': val_loss,
                'score': score,
            }
            #display(val_log)
            if best_val_score < score:
                print("save model weight")
                best_val_preds = val_preds
                best_val_score = score
                torch.save(
                    model.state_dict(), 
                    os.path.join(cfg.EXP_MODEL, f"fold{fold}.pth")
                )

        oof_pred[valid_idx] = best_val_preds.astype(np.float32)
        np.save(os.path.join(cfg.EXP_PREDS, f'oof_pred_fold{fold}.npy'), best_val_preds)
        del model; gc.collect()

    # scoring
    np.save(os.path.join(cfg.EXP_PREDS, 'oof_pred.npy'), oof_pred)
    score = f1_score(np.argmax(oof_pred, axis=1), train['jobflag'], average='macro')
    print('CV:', round(score, 5))
    return score

def main():
    # =====================
    # Main
    # =====================
    # セットアップ
    cfg = setup(config)

    import transformers
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from transformers import AdamW, get_linear_schedule_with_warmup

    # データの読み込み
    train = pd.read_csv(os.path.join(cfg.INPUT, 'train.csv'))
    test = pd.read_csv(os.path.join(cfg.INPUT, 'test.csv'))
    sub = pd.read_csv(os.path.join(cfg.INPUT, 'submit_sample.csv'), header=None)

    # targetの前処理
    train['jobflag'] -= 1

    # tokenizerの読み込み
    cfg.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_PATH)

    # validationデータの設定
    cfg.folds = get_stratifiedkfold(train, 'jobflag', cfg.num_fold, cfg.seed)
    cfg.folds.to_csv(os.path.join(cfg.EXP_PREDS, 'folds.csv'))

    # BERTの学習
    score = training(cfg, train)

    # BERTの推論
    cfg.model_weights = [p for p in sorted(glob(os.path.join(cfg.EXP_MODEL, 'fold*.pth')))]
    sub_pred = inferring(cfg, test)
    sub[1] = np.argmax(sub_pred, axis=1)
    sub[1] = sub[1].astype(int) + 1

    # 提出用ファイル
    sub.to_csv(os.path.join(cfg.EXP_PREDS, 'submission.csv'), index=False, header=False)

if __name__ == "__main__":
    main()
