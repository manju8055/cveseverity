# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
import sys
import streamlit as st
#st.snow()
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.optim import Adam
#from transformers import GPT2Tokenizer
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

from transformers import GPT2Config

from transformers import GPT2Model, GPT2Tokenizer
import nltk
from pyparsing.helpers import removeQuotes
from string import digits
import re

nltk.download('wordnet')


LOGGER = get_logger(__name__)


# Title
st.title("PREDICT THE SEVERITY FOR THE CVE DESCRIPTION")
lst_stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', ' ', '█', '½', 'à', '…', '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in
                    lst_stopwords]
    # print(lst_text)
    for i in range(len(lst_text)):
      remove_digits = str.maketrans('', '', digits)
      # print(li)
      lst_text[i] = lst_text[i].translate(remove_digits)
      # print(li)
    # lst_text = "".join(filter(lambda x: not x.isdigit(), lst_text))
    # text = "".join(x for x in text if x.isalpha())

    # if (re.search(r'\d', lst_text)):
    #     lst_text = re.sub('[0-9]{5,}', '#####', lst_text)
    #     lst_text = re.sub('[0-9]**{4}**', '####', lst_text)
    #     lst_text = re.sub('[0-9]**{3}**', '###', lst_text)
    #     lst_text = re.sub('[0-9]**{2}**', '##', lst_text)
    # for punct in puncts:lst_text if punct in lst_text:lst_text = lst_text.replace(punct, '')
    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]

    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]

    ## back to string from list
    text = " ".join(lst_text)
    return text

# def inference(comment):
#     tokens_test = tokenizer.batch_encode_plus(
#         list([comment]),
#         max_length=75,
#         pad_to_max_length=True,
#         truncation=True,
#         return_token_type_ids=False
#     )
#     test_seq = torch.tensor(tokens_test['input_ids'])
#     test_mask = torch.tensor(tokens_test['attention_mask'])
#     predictions = model(test_seq.to(device), test_mask.to(device))
#     predictions = predictions.detach().cpu().numpy()
#     # print(predictions)
#     predictions = np.argmax(predictions, axis=1)
#     val=int(predictions)
#     # print("INT",val)
#     num=get_GRP_NO(val)

#     return num

# create a GPT2Config object with default parameters
config = GPT2Config()

# set some parameters
# config.vocab_size = 50000
# config.n_embd = 1024
# config.n_head = 16
config.n_layer = 10
# config.max_position_embeddings = 1024



from torch import nn

class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, gpt_model_name:str):
        super(SimpleGPT2SequenceClassifier,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        self.fc1 = nn.Linear(hidden_size*max_seq_len, num_classes)

    def forward(self, input_id, mask):
        """
        Args:
                input_id: encoded inputs ids of sent.
        """
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size,-1))
        return linear_output


def Dataset1(processed_text):

  text=tokenizer(processed_text,padding='max_length',max_length=128,truncation=True,return_tensors="pt")

  return text

def evaluate(model, test_data):

    test_input = Dataset1(test_data)

    # test_dataloader = torch.utils.data.DataLoader(test)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:

        model = model.cuda()

    # Tracking variables
    predictions_labels = []
    # true_labels = []
    # print("HII")

    total_acc_test = 0
    with torch.no_grad():

        # for test_input in test_dataloader:
        if total_acc_test==0:
            # test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            # acc = (output.argmax(dim=1) == test_label).sum().item()
            # total_acc_test += acc
            # add original labels
            # true_labels += test_label.cpu().numpy().flatten().tolist()
            # get predicitons to list
            predictions_labels += output.argmax(dim=1).cpu().numpy().flatten().tolist()
            predd=output.argmax(dim=1).cpu().numpy().flatten().tolist()

    # print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}')

    t1=sev_val[predd[0]]

    st.write("Predicted - ",t1)
    return predd

def model_load():


    model = GPT2Model(config=config)
    model = SimpleGPT2SequenceClassifier(hidden_size=768, num_classes=4, max_seq_len=128, gpt_model_name="gpt2")
    model.load_state_dict(torch.load("gpt2-text-classifier-model_with_84per.pt",map_location=torch.device('cpu')),strict=False)
    model.eval()
    return model

def desc_ip(description):
    if len(description)==0:
      sys.exit("ENTER THE DECRIPTION")
    return description

#description=str(input("Enter the Description : "))
descr= st.text_input("Enter the CVE Description", "")
if(st.button('Submit')):
  description=desc_ip(descr)

  with st.spinner('Wait for it...'):
    model=model_load()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    labels = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    sev_val=["CRITICAL","HIGH","MEDIUM","LOW"]

     #f111 = description.title()
    prepro_descp=utils_preprocess_text(description, flg_stemm=False, flg_lemm=True, lst_stopwords=lst_stopwords)
    print(prepro_descp)
  pred_labels = evaluate(model, prepro_descp)


if __name__ == "__main__":
    pass
