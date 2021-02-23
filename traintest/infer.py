from main import Model, Arg
import pandas as pd
from time import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

import re
import emoji
from soynlp.normalizer import repeat_normalize

finetune_ckpt = './lightning_logs/version_0/checkpoints/epoch=2-step=917.ckpt'
test_path = '../data/inferset.csv'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = Arg()

ckp = torch.load(finetune_ckpt, map_location=torch.device('cpu'))
pretrained_model_config = BertConfig.from_pretrained(
    args.pretrained_model,
    num_labels=ckp['state_dict']['bert.classifier.bias'].shape.numel(),
)
model = BertForSequenceClassification(pretrained_model_config)
model.load_state_dict({k[5:]: v for k, v in ckp['state_dict'].items()})
model.to(device)
model.eval()

def read_data(path):
    if path.endswith('xlsx'):
        return pd.read_excel(path)
    elif path.endswith('csv'):
        return pd.read_csv(path)
    elif path.endswith('tsv') or path.endswith('txt'):
        return pd.read_csv(path, sep='\t')
    else:
        raise NotImplementedError('Only Excel(xlsx)/Csv/Tsv(txt) are Supported')

def preprocess_dataframe(df):
    emojis = ''.join(emoji.UNICODE_EMOJI.keys())
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    tokenizer = BertTokenizer.from_pretrained(
          args.pretrained_tokenizer
        )
    def clean(x):
        x = pattern.sub(' ', x)
        x = url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)
        return x
    result_frame = pd.DataFrame()
    result_frame['txt'] = df['txt'].map(lambda x: tokenizer.encode(
        clean(str(x)),
        padding='max_length',
        max_length=150,
        truncation=True,
    ))
    return result_frame

def test_dataloader(inferset):
    df = preprocess_dataframe(inferset)
    dataset = TensorDataset(
        torch.tensor(df['txt'].to_list(), dtype=torch.long),
    )
    
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

inferset = read_data(test_path)
test_loader = test_dataloader(inferset)
start = time()

with torch.no_grad():
    '''
    훈련한 모델을 탑재해서 하나하나 돌려봤습니다.
    돌려본 결과 모델이 영화인 데이터에서는 상당한 자신감을 보이는 반면,
    영화가 아닌 데이터도 과검하는 현상을 발견했습니다.

    이에 저는 softmax 값에 argmax를 주기보다는, 저만의 threshold를 찾아 적용하는 방식을 택했습니다.
    그 결과 훨씬 정확한 classifcation 값을 얻을 수 있었다고 생각합니다.
    '''
    threshold = 0.95
    scores, preds = [], []

    for i, (x_batch) in enumerate(test_loader):
        prob = model(x_batch[0].to(device)).logits.softmax(dim=1)
        movie_score = round(prob[0][1].item(), 4)
        pred = "movie" if movie_score >= threshold else "notmovie"
        scores.append(movie_score)
        preds.append(pred)
        end = time()
        if i % 20 == 0 :
            print('elapsed time : {:.2f}'.format((end-start)/60), i, movie_score, pred)

inferset['scores'] = scores
inferset['preds'] = preds
# print(inferset)

inferset.to_csv('result.csv', encoding = 'utf-8-sig')
