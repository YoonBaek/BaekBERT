from main import Model, Arg
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from transformers import BertForSequenceClassification, BertTokenizer

args = Arg()
model = Model(options = args).load_from_checkpoint('../lightning_logs/version_8/checkpoints/epoch=2-step=458.ckpt',
        options = args)
model.freeze()

class Predictor:
    def __init__(self, config):
        self.device = "cuda:0"
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrained_tokenizer)
        self.model = model

    def predict(self, payload):
        inputs = self.tokenizer.encode_plus(payload["text"], return_tensors="pt")
        predictions = self.model(**inputs)[0]
        if (predictions[0] > predictions[1]):
          return {"class": "unacceptable"}
        else:
          return {"class": "acceptable"}