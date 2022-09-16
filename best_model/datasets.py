import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler

import transformers
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

# =====================
# Dataset & Model
# =====================
class BERTDataset(Dataset):
    def __init__(self, cfg, texts, labels=None):
        self.cfg = cfg
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        inputs = self.prepare_input(self.cfg, self.texts[index])
        if self.labels is not None:
            label = torch.tensor(self.labels[index], dtype=torch.int64)
            return inputs, label
        else:
            return inputs
    
    @staticmethod
    def prepare_input(cfg, text):
        inputs = cfg.tokenizer(
            text,
            add_special_tokens=True,
            max_length=cfg.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=False,
        )
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
        return inputs


class BERTModel(nn.Module):
    def __init__(self, cfg, criterion=None):
        super().__init__()
        self.cfg = cfg
        self.criterion = criterion
        self.config = AutoConfig.from_pretrained(
            cfg.MODEL_PATH,
            output_hidden_states=True
        )
        self.backbone = AutoModel.from_pretrained(
            cfg.MODEL_PATH, 
            config=self.config
        )
        self.fc = nn.Sequential(
            nn.Linear(self.config.hidden_size, 4),
        )
    
    def forward(self, inputs, labels=None):
        outputs = self.backbone(**inputs)["last_hidden_state"]
        outputs = outputs[:, 0, :]
        if labels is not None:
            logits = self.fc(outputs)
            loss = self.criterion(logits, labels)
            return logits, loss
        else:
            logits = self.fc(outputs)
            return logits
