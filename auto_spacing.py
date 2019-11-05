from keras.models import load_model
# from pykospacing.embedding_maker import load_vocab, encoding_and_padding
import os
import pkg_resources
import re
import warnings
from keras.preprocessing import sequence
import json
import numpy as np


__all__ = ['spacing', ]
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #tensorflow 오류시 작성한 코드

def load_embedding(embeddings_file):
    return(np.load(embeddings_file))


def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        data = json.loads(f.read())
    word2idx = data
    idx2word = dict([(v, k) for k, v in data.items()])
    return word2idx, idx2word


def encoding_and_padding(word2idx_dic, sequences, **params):
    """
    1. making item to idx
    2. padding
    :word2idx_dic
    :sequences: list of lists where each element is a sequence
    :maxlen: int, maximum length
    :dtype: type to cast the resulting sequence.
    :padding: 'pre' or 'post', pad either before or after each sequence.
    :truncating: 'pre' or 'post', remove values from sequences larger than
        maxlen either in the beginning or in the end of the sequence
    :value: float, value to pad the sequences to the desired value.
    """
    seq_idx = [[word2idx_dic.get(a, word2idx_dic['__ETC__']) for a in i] for i in sequences]
    params['value'] = word2idx_dic['__PAD__']
    return(sequence.pad_sequences(seq_idx, **params))

model_path = os.path.join("E:\Programming\python\창회선배스터디\Auto_spacing","kospacing")
dic_path =os.path.join("E:\Programming\python\창회선배스터디\Auto_spacing", 'c2v.dic')
model = load_model(model_path)
model._make_predict_function()
w2idx, _ = load_vocab(dic_path)


class pred_spacing:
    def __init__(self, model, w2idx):
        self.model = model
        self.w2idx = w2idx
        self.pattern = re.compile(r'\s+')

    def get_spaced_sent(self, raw_sent):
        raw_sent_ = "«" + raw_sent + "»"
        raw_sent_ = raw_sent_.replace(' ', '^')
        sents_in = [raw_sent_, ]
        mat_in = encoding_and_padding(word2idx_dic=self.w2idx, sequences=sents_in, maxlen=200, padding='post', truncating='post')
        results = self.model.predict(mat_in)
        mat_set = results[0, ]
        preds = np.array(['1' if i > 0.5 else '0' for i in mat_set[:len(raw_sent_)]])
        return self.make_pred_sents(raw_sent_, preds)

    def make_pred_sents(self, x_sents, y_pred):
        res_sent = []
        for i, j in zip(x_sents, y_pred):
            if j == '1':
                res_sent.append(i)
                res_sent.append(' ')
            else:
                res_sent.append(i)
        subs = re.sub(self.pattern, ' ', ''.join(res_sent).replace('^', ' '))
        subs = subs.replace('«', '')
        subs = subs.replace('»', '')
        return subs


pred_spacing = pred_spacing(model, w2idx)


def spacing(sent):
    if len(sent) > 198:
        warnings.warn('One sentence can not contain more than 198 characters. : {}'.format(sent))
    spaced_sent = pred_spacing.get_spaced_sent(sent)
    return(spaced_sent.strip())