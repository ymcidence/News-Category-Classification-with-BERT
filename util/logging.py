import logging
from pathlib import Path
from general import ROOT
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_PATH = Path(os.path.join(ROOT, 'data/news_cat/'))
DATA_PATH2 = Path(os.path.join(ROOT, 'data/partisan/'))
DATA_PATH.mkdir(exist_ok=True)

PATH = Path(os.path.join(ROOT, 'data/news_cat/tmp'))
PATH2 = Path(os.path.join(ROOT, 'data/partisan/tmp'))
PATH.mkdir(exist_ok=True)

CLAS_DATA_PATH = PATH / 'class'
CLAS_DATA_PATH2 = PATH2 / 'class'
CLAS_DATA_PATH.mkdir(exist_ok=True)

model_state_dict = None

BERT_PRETRAINED_PATH = Path('uncased_L-12_H-768_A-12')
args = {
    "train_size": -1,
    "val_size": -1,
    "full_data_dir": DATA_PATH,
    "data_dir": PATH,
    "task_name": "news_cat_label",
    "no_cuda": False,
    "bert_model": 'bert-base-uncased',
    "output_dir": CLAS_DATA_PATH / 'output',
    "max_seq_length": 100,
    "do_train": True,
    "do_eval": True,
    "do_lower_case": True,
    "train_batch_size": 32,
    "eval_batch_size": 32,
    "learning_rate": 3e-5,
    "num_train_epochs": 50,
    "warmup_proportion": 0.1,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": False,
    "loss_scale": 128
}
args2 = {
    "train_size": -1,
    "val_size": -1,
    "full_data_dir": DATA_PATH2,
    "data_dir": PATH2,
    "task_name": "news_cat_label",
    "no_cuda": False,
    "bert_model": 'bert-base-uncased',
    "output_dir": CLAS_DATA_PATH / 'output',
    "max_seq_length": 100,
    "do_train": True,
    "do_eval": True,
    "do_lower_case": True,
    "train_batch_size": 32,
    "eval_batch_size": 32,
    "learning_rate": 3e-5,
    "num_train_epochs": 50,
    "warmup_proportion": 0.1,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": False,
    "loss_scale": 128
}