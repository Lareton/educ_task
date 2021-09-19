# импортируем нужные библиотеки

from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import torch
import json


class BertClassifier:
    """ класс для инициализации и получения предсказаний дообученной ruBert модели """
    def __init__(self, model_path):
        """
        model_path: путь до весов предобученной модели;
        """
        self.model = torch.load(model_path)
        self.tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_len = 512
        self.class_decoder = {0: "спорт", 1: "музыка", 2: "литература", 3: "животные"}
    
    def predict(self, text: str) -> int:
        """ 
        метод для получения предсказания модели на тексте
        Принимает на вход: str, текст;
        Возвращает: str, предсказанный класс.
        """
        text = " ".join(text.split(" ")[:min(len(text), 513)]) # обрежим текст до первых 512 слов
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        out = {
              'text': text,
              'input_ids': encoding['input_ids'].flatten(),
              'attention_mask': encoding['attention_mask'].flatten()
          }
        
        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)
        
        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )
        
        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

        return self.class_decoder[prediction]
