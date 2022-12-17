import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import DataCollatorWithPadding, logging

from tqdm import tqdm
from utils import multi_label_metrics, preprocess_data
import constants


os.environ["TOKENIZERS_PARALLELISM"] = 'false'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
logging.set_verbosity_error()



def evaluate(opt):    
    tokenizer = AutoTokenizer.from_pretrained(opt['ckpt_path'])
    dataset = load_dataset('csv', data_files={'test': opt['test_dataset_path']})
    dataset = dataset.map(preprocess_data,
                                  batched=True,
                                  remove_columns=dataset['test'].column_names,
                                  fn_kwargs={'tokenizer': tokenizer,
                                             'labels': list(constants.ID2LABEL_EN.values())
                                             }
                                  )
    dataset.set_format('torch')
    dataloader = torch.utils.data.DataLoader(dataset['test'],
                                             batch_size=opt['batch_size'],
                                             shuffle=False,
                                             num_workers=opt['num_workers'],
                                             collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
                                             )
    
    scores = {'micro_f1': [],
            'roc_auc': [],
            'accuracy': []
            }
    device = torch.device(opt['device'])
    model = AutoModelForSequenceClassification.from_pretrained(opt['ckpt_path']).to(device)
    
    model.eval()
    for data in tqdm(dataloader, total=len(dataloader), ncols=100):
        inputs = {'input_ids': data['input_ids'].to(device),
                    'token_type_ids': data['token_type_ids'].to(device),
                    'attention_mask': data['attention_mask'].to(device)}
        labels = data['labels']
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu()
        
        score = multi_label_metrics(logits, labels)
        scores['micro_f1'].append(score['f1'])
        scores['roc_auc'].append(score['roc_auc'])
        scores['accuracy'].append(score['accuracy'])

    micro_f1 = np.mean(scores['micro_f1'])
    roc_auc = np.mean(scores['roc_auc'])
    accuracy = np.mean(scores['accuracy'])
    print(f'micro_f1: {micro_f1:.4f}, roc_acu: {roc_auc:.4f}, accuracy: {accuracy:.4f}')
    


if __name__ == '__main__':
    opt = {'ckpt_path': 'weights/20221214T20-40-08/checkpoint-4170',
           'test_dataset_path': 'data/preprocess/kfsc-multi-label-classification-test.csv',
           'device': 'cuda:0',
           'batch_size': 64,
           'num_workers': 4,
           }
    
    evaluate(opt)