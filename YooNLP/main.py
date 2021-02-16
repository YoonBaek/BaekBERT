'''
-------------------------------------------------------------------------------------------------------
author : 백승윤
Reference : https://github.com/Beomi/KcBERT

BERT를 기반으로 한국어 댓글에 대해 pretrain 된 모델이라 본 task에 적합하다고 판단했습니다.
PyTorch Lightning으로 작성됐기 때문에 Vision이 아닌 nlp task에서도
ddp나 mixed precision과 같은 대용량 훈련 기법을 안정적으로 구동할 수 있다는 장점도 있어 reference로 활용하기로 했습니다.

Baseline은 huggingface의 Transformers git에서 배포된 BERT 모델입니다.
본 코드는 Multi GPU 환경에서 구동했습니다.

구동하실 때에는
cpu workers = 0으로 구동하시길 추천드립니다.
-------------------------------------------------------------------------------------------------------

나날이 생겨나는 신조어, 새로운 유형의 변칙 단어, 오타를 일일히 cleansing하기는 어렵습니다.
기본적인 cleansing만 진행하고도 강건함을 보일 수 있는 모델이 있으면 어떨까 하는 생각에
미리 comment로 훈련된 모델을 불러오고, 그 모델을 영화 여부 분류에 맞게 down stream하는 훈련을 진행했습니다.

kcbert-base, large 모델은 제가 아는 한에서 nsmc SOTA를 기록한 pretrain 모델이라 골랐습니다.

nsmc셋 + 크롤링 데이터로 down stream한 모델로 생성된 checkpoint로 제시받은 dataset을 분류하고,
특정 score 미만의 confidence를 보이는 데이터만 따로 확인하는 process를 구상해봤습니다.

Downstream 방식은
BERT: Pre-trainig of Deep Bidirectional Transformers for Language Understanding에서 제시된 방식을 따랐습니다.
'''
import os
import pandas as pd

from pprint import pprint

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingWarmRestarts

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import BertForSequenceClassification, BertTokenizer, AdamW

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# text compile 및 cleaning
import re
import emoji
from soynlp.normalizer import repeat_normalize

# 학습에 사용될 Args를 정의합니다.
class Arg:
    random_seed: int = 33  # Random Seed
    # pretrain된 모델을 불러옵니다.
    pretrained_model: str = 'beomi/kcbert-large' # korean comment pretrained set load
    pretrained_tokenizer: str = 'beomi/kcbert-base'
    auto_batch_size: str = 'power'  # by 최적의 batch를 찾아주는 기능이나, 여기서는 미사용. batch_size 입력시 자동으로 off
    batch_size: int = 16
    lr: float = 5e-6  
    epochs: int = 5 
    max_length: int = 150  
    report_cycle: int = 100 # Report (Train Metrics) Cycle
    train_data_path: str = "train_dataset.csv"  # Train Dataset file 
    val_data_path: str = "valid_dataset.csv"  # Validation Dataset file 
    cpu_workers: int = 0 # Multi cpu workers
    test_mode: bool = False  # Test Mode enables `fast_dev_run`
    optimizer: str = 'AdamW'  # AdamW vs AdamP
    lr_scheduler: str = 'cos'  # ExponentialLR vs CosineAnnealingWarmRestarts
    fp16: bool = False  # Enable train on FP16
    tpu_cores: int = 0  # Enable TPU with 1 core or 8 cores

args = Arg()

# args.tpu_cores = 8  # Enables TPU
args.fp16 = False  # Enables GPU FP16
# args.batch_size = 16  # Force setup batch_size

class Model(LightningModule):
    def __init__(self, options):
        super().__init__()
        self.args = options
        self.bert = BertForSequenceClassification.from_pretrained(self.args.pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(
            self.args.pretrained_tokenizer
            if self.args.pretrained_tokenizer
            else self.args.pretrained_model
        )

    def forward(self, **kwargs):
        return self.bert(**kwargs)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        output = self(input_ids=data, labels=labels)

        # Transformers 4.0.0+
        loss = output.loss
        logits = output.logits
        
        preds = logits.argmax(dim=-1)

        y_true = labels.cpu().numpy()
        y_pred = preds.cpu().numpy()

        # Acc, Precision, Recall, F1
        metrics = [
            metric(y_true=y_true, y_pred=y_pred)
            for metric in
            (accuracy_score, precision_score, recall_score, f1_score)
        ]

        tensorboard_logs = {
            'train_loss': loss.cpu().detach().numpy().tolist(),
            'train_acc': metrics[0],
            'train_precision': metrics[1],
            'train_recall': metrics[2],
            'train_f1': metrics[3],
            # 'lr' : optimizer.state_dict()['param_groups']['lr']
        }
        if (batch_idx % self.args.report_cycle) == 0:

            pprint(tensorboard_logs)
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        output = self(input_ids=data, labels=labels)

        # Transformers 4.0.0+
        loss = output.loss
        logits = output.logits

        preds = logits.argmax(dim=-1)

        y_true = list(labels.cpu().numpy())
        y_pred = list(preds.cpu().numpy())

        return {
            'loss': loss,
            'y_true': y_true,
            'y_pred': y_pred,
        }

    def validation_epoch_end(self, outputs):
        loss = torch.tensor(0, dtype=torch.float)
        for i in outputs:
            loss += i['loss'].cpu().detach()
        _loss = loss / len(outputs)

        loss = float(_loss)
        y_true = []
        y_pred = []

        for i in outputs:
            y_true += i['y_true']
            y_pred += i['y_pred']

        # Acc, Precision, Recall, F1
        metrics = [
            metric(y_true=y_true, y_pred=y_pred)
            for metric in
            (accuracy_score, precision_score, recall_score, f1_score)
        ]

        tensorboard_logs = {
            'val_loss': loss,
            'val_acc': metrics[0],
            'val_precision': metrics[1],
            'val_recall': metrics[2],
            'val_f1': metrics[3],
        }

        print()
        pprint(tensorboard_logs)
        return {'loss': _loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        if self.args.optimizer == 'AdamW':
            optimizer = AdamW(self.parameters(), lr=self.args.lr)
        elif self.args.optimizer == 'AdamP':
            from adamp import AdamP
            optimizer = AdamP(self.parameters(), lr=self.args.lr)
        else:
            raise NotImplementedError('Only AdamW and AdamP is Supported!')
        if self.args.lr_scheduler == 'cos':
            scheduler = {'scheduler' : CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2)}
        elif self.args.lr_scheduler == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.5)
        else:
            raise NotImplementedError('Only cos and exp lr scheduler is Supported!')
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def read_data(self, path):
        if path.endswith('xlsx'):
            return pd.read_excel(path)
        elif path.endswith('csv'):
            return pd.read_csv(path)
        elif path.endswith('tsv') or path.endswith('txt'):
            return pd.read_csv(path, sep='\t')
        else:
            raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')

    def preprocess_dataframe(self, df):
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        url_pattern = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

        def clean(x):
            x = pattern.sub(' ', x)
            x = url_pattern.sub('', x)
            x = x.strip()
            x = repeat_normalize(x, num_repeats=2)
            return x

        df['txt'] = df['txt'].map(lambda x: self.tokenizer.encode(
            clean(str(x)),
            padding='max_length',
            max_length=self.args.max_length,
            truncation=True,
        ))
        return df

    def train_dataloader(self):
        df = self.read_data(self.args.train_data_path)
        df = self.preprocess_dataframe(df)

        dataset = TensorDataset(
            torch.tensor(df['txt'].to_list(), dtype=torch.long),
            torch.tensor(df['label'].to_list(), dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size or self.batch_size,
            shuffle=True,
            num_workers=self.args.cpu_workers,
        )

    def val_dataloader(self):
        df = self.read_data(self.args.val_data_path)
        df = self.preprocess_dataframe(df)

        dataset = TensorDataset(
            torch.tensor(df['txt'].to_list(), dtype=torch.long),
            torch.tensor(df['label'].to_list(), dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size or self.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )
def main():
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args.random_seed)
    seed_everything(args.random_seed)
    model = Model(args)
cd
    print(":: Start Training ::")
    trainer = Trainer(
        max_epochs=args.epochs,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        auto_scale_batch_size=args.auto_batch_size if args.auto_batch_size and not args.batch_size else False,
        # For GPU Setup
        deterministic=torch.cuda.is_available(),
        # gpus=-1 if torch.cuda.is_available() else None,
        gpus=-1 if torch.cuda.is_available() else None,
        precision=16 if args.fp16 else 32
        # For TPU Setup
        # tpu_cores=args.tpu_cores if args.tpu_cores else None,
    )
    trainer.fit(model)
if __name__ == '__main__':
    # 이 파일을 구동할 때만 main함수가 작동하도록 따로 분리를 해줘야만 훈련이 제대로 이뤄집니다.
    main()