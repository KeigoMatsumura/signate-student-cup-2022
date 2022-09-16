import os
import torch

AUTHOR = "keigo"

# 出力フォルダ名
NAME = "Exp017-deberta-v3-large-epoch10-batch8-cv4"

# 学習するモデルの読み込み
# https://huggingface.co/ からモデルのパスを指定
# 例えば, "microsoft/deberta-base"
MODEL_PATH = "microsoft/deberta-v3-large"

# ベースとなるディレクトリパスの指定
COLAB_PATH = "SIGNATE2022" 
DRIVE_PATH = os.path.join(COLAB_PATH, AUTHOR)

# クラスの重み
num_of_classes = 1516
class_weights = torch.tensor([num_of_classes/505, num_of_classes/468, num_of_classes/455, num_of_classes/88]).cuda()

# シード値
seed = 42
    
# cross-validaitonの分割数
num_fold = 4
# 学習するfold
trn_fold = [0, 1, 2, 3]
    
# batct_sizeの設定
batch_size = 8
    
# epoch数の設定
n_epochs = 10
    
# トークン数の最大の長さの設定
max_len = 128

# 学習率の設定
lr = 2e-5

# optimizer等の設定
weight_decay = 2e-5
beta = (0.9, 0.98)
num_warmup_steps_rate = 0.01
clip_grad_norm = None
gradient_accumulation_steps = 1
num_eval = 1


# net = torch.nn.DataParallel(net, device_ids=[0, 1])


def setup(cfg):
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set dirs
    cfg.DRIVE = cfg.DRIVE_PATH
    cfg.EXP = (cfg.NAME if cfg.NAME is not None 
        else requests.get('http://172.28.0.2:9000/api/sessions').json()[0]['name'][:-6]
    )
    cfg.INPUT = os.path.join(cfg.DRIVE, 'Input')
    cfg.OUTPUT = os.path.join(cfg.DRIVE, 'Output')
    cfg.DATASET = os.path.join(cfg.DRIVE, 'Dataset')

    cfg.OUTPUT_EXP = os.path.join(cfg.OUTPUT, cfg.EXP) 
    cfg.EXP_MODEL = os.path.join(cfg.OUTPUT_EXP, 'model')
    cfg.EXP_FIG = os.path.join(cfg.OUTPUT_EXP, 'fig')
    cfg.EXP_PREDS = os.path.join(cfg.OUTPUT_EXP, 'preds')

    # make dirs
    for d in [cfg.INPUT, cfg.EXP_MODEL, cfg.EXP_FIG, cfg.EXP_PREDS]:
        os.makedirs(d, exist_ok=True)
    return cfg
