import os
import pandas as pd
from glob import glob

from pprint import pprint
import torch
from pytorch_lightning import Trainer
from model import FrownSentenceClassifier


def evaluate(args):
    trainer = Trainer(fast_dev_run=args['test_mode'],
                      num_sanity_val_steps=None if args['test_mode'] else 0,
                      deterministic=torch.cuda.is_available(),
                      accelerator='gpu',
                      devices=[args['gpu']] if torch.cuda.is_available() else None,  # 0번 idx GPU  사용
                      precision=16 if args['fp16'] and torch.cuda.is_available() else 32
                      )

    model = FrownSentenceClassifier(**args)
    model.freeze()
    model.eval()
    trainer.test(model=model,
                ckpt_path=args['test_ckpt_path']
                )


if __name__ == '__main__':
    args = {
        'random_seed': 1031, # Random Seed
        'pretrained_model': 'beomi/KcELECTRA-base-v2022',  # Transformers PLM name
        'pretrained_tokenizer': 'beomi/KcELECTRA-base-v2022',  # Optional, Transformers Tokenizer Name. Overrides `pretrained_model`
        'batch_size': 128,
        'lr': 5e-6,  # Starting Learning Rate
        'epochs': 20,  # Max Epochs
        'max_length': 60,  # Max Length input size
        'test_data_path': "data/preprocess/fsc-test-v2.csv",  # Test Dataset file
        'test_mode': True,  # Test Mode enables `fast_dev_run`
        'optimizer': 'AdamW',  # AdamW vs AdamP
        'lr_scheduler': 'exp',  # ExponentialLR vs CosineAnnealingWarmRestarts
        'fp16': True,  # Enable train on FP16(if GPU)
        'tpu_cores': 0,  # Enable TPU with 1 core or 8 cores
        'cpu_workers': 4,
        'gpu': 0,
        'test_ckpt_path': '/home/ubuntu/JupyterProjects/limkaram/frown-sentence-classifier/lightning_logs/version_20/checkpoints/epoch4-val_loss0.3918-val_micro_f10.8594-val_macro_f10.8337-val_acc0.8594.ckpt',
        'num_labels': 8
    }

    evaluate(args)
    
