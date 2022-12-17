import os
import re
import emoji
import torch
import numpy as np
from datetime import datetime
from soynlp.normalizer import repeat_normalize
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction


def clean(text):
    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
    text = pattern.sub(' ', text)
    text = emoji.replace_emoji(text, replace='') #emoji 삭제
    text = url_pattern.sub('', text)
    text = text.strip()
    text = repeat_normalize(text, num_repeats=2)
    
    return text

def make_current_datetime_dir(path):
    now = datetime.now().strftime(r'%Y%m%dT%H-%M-%S')
    make_dir_path = os.path.join(path, now)
    os.mkdir(make_dir_path)
    
    return make_dir_path
        

def preprocess_data(examples, tokenizer, labels):
    # take a batch of texts
    sentences = [clean(sentence) for sentence in examples['document']]  # KcELECTRA 사전 학습시 사용한 정제 적용
    
    # encode them
    # encoding = tokenizer(sentences, padding='max_length', truncation=True, max_length=80)
    
    '''
    Trainer에서 data_collector 사용으로
    padding, max_length 옵션을 지정해주지 않아도됨.
    data_collector에서 각 미니 배치에 포함된 sequence 중 가장 긴 sequence를 기준으로 나머지 문장을 padding 함
    '''
    encoding = tokenizer(sentences, truncation=True)
    
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(sentences), len(labels)))
    
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding['labels'] = labels_matrix.tolist()
    
    return encoding


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result


if __name__ == '__main__':
    make_current_datetime_dir('/home/ubuntu/JupyterProjects/limkaram/korean-frown-sentence-classifier/weights')