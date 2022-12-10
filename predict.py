import time
import torch
import pandas as pd
import re
import emoji
from soynlp.normalizer import repeat_normalize
from model import FrownSentenceClassifier


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

def infer(text):
    print(f'input text: {text}')
    label2kor = {0: '출신차별',
            1: '외모차별',
            2: '정치성향차별',
            3: '혐오욕설',
            4: '연령차별',
            5: '성/가족차별',
            6: '종교차별',
            7: '해당사항없음'
            }
    
    label2en = {0: 'Origin',
            1: 'Physical',
            2: 'Politics',
            3: 'Profanity',
            4: 'Age',
            5: 'Gender/Family',
            6: 'Religion',
            7: 'Not Hate Speech'
            }
    
    global model
    global device
    text = clean(text)
    
    infer_stime = time.time()
    tokens = model.tokenizer(text, return_tensors='pt').to(device)
    logits = model(**tokens).logits
    infer_etime = time.time()
    preds = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    labels = [label for label in label2kor.values()]
    df = pd.DataFrame({'Label': labels, 'Probability': preds.reshape(-1,)})
    print(f'inference elapsed time: {infer_etime - infer_stime:.3f}s')
    
    return df
    

if __name__ == '__main__':
    ckpt_path = '/Users/limkaram/PersonalSpace/frown-sentence-classifier/epoch4-val_loss0.3918-val_micro_f10.8594-val_macro_f10.8337-val_acc0.8594.ckpt'
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f'set device: {device}')
    model = FrownSentenceClassifier.load_from_checkpoint(ckpt_path).to(device)
    model.eval()
    model.freeze()
    
    texts = ['까람이 개같이생겼네..']
    for text in texts:
        result_df = infer(text)
        print(result_df)
        print()

    