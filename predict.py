import time
import torch
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import constants
from utils import clean



def infer(sentences):
    global model
    global tokenizer
    global device
    id2label = constants.ID2LABEL_KOR
    results = []
    
    for sentence in sentences:
        sentence = clean(sentence)

        infer_stime = time.time()
        encoding = tokenizer(sentence, return_tensors='pt').to(device)
        outputs = model(**encoding)
        logits = outputs.logits
        sigmoid = torch.nn.Sigmoid()
        preds = sigmoid(logits.squeeze())
        infer_etime = time.time()
        
        result = {'문장': sentence,
                  '추론시간': infer_etime - infer_stime
                  }
        
        for id, label in id2label.items():
            prob = preds[id].item()
            result[label] = prob
        
        results.append(result)
        
    results = pd.DataFrame(results)
    
    return results


if __name__ == '__main__':
    ckpt_path = 'weights/20221214T20-40-08/checkpoint-4170'
    device = 'cuda:0'
    model = AutoModelForSequenceClassification.from_pretrained(ckpt_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    
    sentences = ['태극기 늙은 멍멍이들 오래 살아야 민주당 장기 집권하지ㅋㅋ']
    
    model.eval()
    ret = infer(sentences)
    print(ret)
    
    