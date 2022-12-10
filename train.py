import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything

from model import FrownSentenceClassifier


def train(args):
    checkpoint_callback = ModelCheckpoint(
        filename='epoch{epoch}-val_loss{val_loss:.4f}-val_micro_f1{val_micro_f1:.4f}-val_macro_f1{val_macro_f1:.4f}-val_acc{val_acc:.4f}',
        monitor='val_loss',
        save_top_k=10,
        mode='min',
        auto_insert_metric_name=False,
    )
    
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args['random_seed'])
    seed_everything(args['random_seed'])
    model = FrownSentenceClassifier(**args)

    print(":: Start Training ::")
    trainer = Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=args['epochs'],
        fast_dev_run=args['test_mode'],
        num_sanity_val_steps=None if args['test_mode'] else 0,
        deterministic=torch.cuda.is_available(),
        accelerator='gpu',
        devices=[args['gpu']] if torch.cuda.is_available() else None,  # 0번 idx GPU  사용
        precision=16 if args['fp16'] and torch.cuda.is_available() else 32,
    )
    trainer.fit(model)
    

if __name__ == '__main__':
    args = {
        'random_seed': 1031, # Random Seed
        'pretrained_model': 'beomi/KcELECTRA-base-v2022',  # Transformers PLM name
        'pretrained_tokenizer': 'beomi/KcELECTRA-base-v2022',  # Optional, Transformers Tokenizer Name. Overrides `pretrained_model`
        'batch_size': 128,
        'lr': 5e-6,  # Starting Learning Rate
        'epochs': 6,  # Max Epochs
        'max_length': 60,  # Max Length input size
        'train_data_path': "data/preprocess/fsc-train-v2.csv",  # Train Dataset file 
        'val_data_path': "data/preprocess/fsc-val-v2.csv",  # Validation Dataset file 
        'test_data_path': "data/preprocess/fsc-test-v2.csv",  # Test Dataset file
        'test_mode': False,  # Test Mode enables `fast_dev_run`
        'optimizer': 'AdamW',  # AdamW vs AdamP
        'lr_scheduler': 'exp',  # ExponentialLR vs CosineAnnealingWarmRestarts
        'fp16': True,  # Enable train on FP16(if GPU)
        'tpu_cores': 0,  # Enable TPU with 1 core or 8 cores
        'cpu_workers': 4,
        'gpu': 0,
        'num_labels': 8
    }
    
    train(args)
