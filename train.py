import os
import constants
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer, logging
from utils import clean, make_current_datetime_dir, compute_metrics, preprocess_data


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
logging.set_verbosity_error()


def train(opt):
    tokenizer = AutoTokenizer.from_pretrained(opt['pretrained_tokenizer'])
    id2label = constants.ID2LABEL_EN
    label2id = {v: k for k, v in id2label.items()}
    labels = list(label2id.keys())
    dataset = load_dataset('csv', data_files={'train': opt['train_dataset_path'],
                                              'val': opt['val_dataset_path']
                                              }
                           )
    dataset = dataset.map(preprocess_data,
                                  batched=True,
                                  remove_columns=dataset['train'].column_names,
                                  fn_kwargs={'tokenizer': tokenizer,
                                             'labels': labels
                                             }
                                  )
    dataset.set_format('torch')
    model = AutoModelForSequenceClassification.from_pretrained(opt['pretrained_model'], 
                                                            problem_type=opt['problem_type'],
                                                            num_labels=len(labels),
                                                            id2label=id2label,
                                                            label2id=label2id)
    args = TrainingArguments(output_dir=make_current_datetime_dir(opt['output_dir']),
                            evaluation_strategy=opt['evaluation_strategy'],
                            save_strategy=opt['save_strategy'],
                            learning_rate=opt['learning_rate'],
                            per_device_train_batch_size=opt['per_device_train_batch_size'],
                            per_device_eval_batch_size=opt['per_device_eval_batch_size'],
                            num_train_epochs=opt['num_train_epochs'],
                            weight_decay=opt['weight_decay'],
                            load_best_model_at_end=opt['load_best_model_at_end'],
                            metric_for_best_model=opt['metric_for_best_model'],
                            seed=opt['seed'],
                            dataloader_num_workers=opt['dataloader_num_workers'],
                            no_cuda=opt['no_cuda']
                            )
    trainer = Trainer(args=args,
                      model=model,
                      tokenizer=tokenizer,
                      train_dataset=dataset['train'],
                      eval_dataset=dataset['val'],
                      compute_metrics=compute_metrics,
                      data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
                      )
    
    trainer.train()
    

if __name__ == '__main__':
    opt = {'pretrained_model': 'beomi/KcELECTRA-base-v2022',
           'pretrained_tokenizer': 'beomi/KcELECTRA-base-v2022',
           'problem_type': 'multi_label_classification',
           'train_dataset_path': 'data/preprocess/kfsc-multi-label-classification-train.csv',
           'val_dataset_path': 'data/preprocess/kfsc-multi-label-classification-val.csv',
           'output_dir': 'weights/',
           'metric_for_best_model': 'f1',
           'evaluation_strategy': 'epoch',
           'save_strategy': 'epoch',
           'seed': 1031,
           'no_cuda': False,
           'learning_rate': 5e-6,
           'per_device_train_batch_size': 16,
           'per_device_eval_batch_size': 16,
           'num_train_epochs': 10,
           'weight_decay': 0.01,
           'dataloader_num_workers': 4,
           'load_best_model_at_end': False,
           }
    
    train(opt)