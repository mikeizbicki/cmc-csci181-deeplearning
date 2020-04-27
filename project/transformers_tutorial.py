#!/usr/bin/python3

# these lines prevent lots of warnings from being displayed
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import the deep learning libraries
import torch
import transformers

# load the model
#checkpoint_name = 'bert-base-uncased'
checkpoint_name = 'bert-base-multilingual-uncased'
tokenizer = transformers.BertTokenizer.from_pretrained(checkpoint_name)
bert = transformers.BertModel.from_pretrained(checkpoint_name)

# sample data
lines = [
    'The coronavirus pandemic has taken over the world.',    # English
    'La pandemia de coronavirus se ha apoderado del mundo.', # Spanish
    'La pandemia di coronavirus ha conquistato il mondo.',   # Italian
    'Capta est coronavirus pandemic in orbe terrarum.',      # Latin
    'đại dịch coronavirus đã chiếm lĩnh thế giới.',          # Vietnamese
    'пандемия коронавируса захватила мир.',                  # Russian
    'سيطر وباء الفيروس التاجي على العالم.',                  # Arabic
    'מגיפת הנגיף השתלט על העולם.',                           # Hebrew
    '코로나 바이러스 전염병이 세계를 점령했습니다.',         # Korean
    '冠狀病毒大流行已席捲全球。',                            # Chinese (simplified)
    '冠状病毒大流行已经席卷全球。',                          # Chinese (traditional)
    'コロナウイルスのパンデミックが世界を席巻しました。',    # Japanese
    ]

for line in lines:
    tokens = tokenizer.tokenize(line)
    print("tokens=",tokens)
crash

# generates 1-hot encodings of the lines
max_length = 64
encodings = []
#lines = lines[0:1]
for line in lines:
    encoding = tokenizer.encode_plus(
        line,
        #add_special_tokens = True,
        max_length = max_length,
        pad_to_max_length = True,
        #return_attention_mask = True,
        return_tensors = 'pt',
        )
    #print("encoding.keys()=",encoding.keys())
    #print("encoding['input_ids'].shape=",encoding['input_ids'].shape)
    #print("encoding['input_ids']=",encoding['input_ids'])
    encodings.append(encoding)

input_ids = torch.cat([encoding['input_ids'] for encoding in encodings ],dim=0)
#attention_mask = torch.cat([ encoding['attention_mask'] for encoding in encodings ],dim=0)

import datetime
for i in range(10):
    print(datetime.datetime.now())
    last_layer,embedding = bert(input_ids) #, attention_mask)
    print("last_layer.shape=",last_layer.shape)
    print("embedding.shape=",embedding.shape)
    crash


class BertFineTuning(nn.Module):
    def __init__(self):
        super().__init__()
        #self.bert = transformers.BertModel.from_pretrained(checkpoint_name)
        self.fc = nn.Linear(768,num_classes)

    def forward(self,x):
        #last_layer,embedding = self.bert(x) 
        last_layer,embedding = bert(x) 
        embedding = torch.mean(last_layer,dim=1)
        out = self.fc(embedding)
        return out
