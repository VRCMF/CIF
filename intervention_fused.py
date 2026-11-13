import os, re, json, pickle
import torch
import torch.nn.functional as F
import json
import numpy as np

from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap,
)
from causal_trace import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)
from util.rome.dsets import KnownsDataset
from util.rome.util import Trace, TraceDict
import ipdb
import logging
import time
import matplotlib.pyplot as plt
from numpy import *

import torch
import numpy as np
from typing import List
from util.xcopa_eval import run_xcopa
from util.xcsr_eval import run_xcsr
from util.xwinograd_eval import run_xwinograd
from util.xnli_eval import run_xnli
from util.xstorycloze_eval import run_xstorycloze
from util.mgsm_eval import run_mgsm
from util.mzsre_eval import run_mzsre
from util.wmt_eval import run_wmt
from util.flores_eval import run_flores
import util.globals as util_globals
import argparse
import sys
import random
from utils import get_out_mean, get_out

torch.set_grad_enabled(False)

DEFAULT_MODEL_NAME = "bigscience/bloomz-7b1-mt"
mt = None
LAYERS = None



def trace_with_patch_discrimination(
    model,  
    inp,  #
    states_to_patch,  
    new_mlp_up,
    question,
    answer,
):
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer not in patch_spec:
            return x
        h = untuple(x)
        for t in patch_spec[layer]:
            layer_match = re.search(r"(?:transformer\.h|model\.layers|gpt_neox\.layers)\.(\d+)", layer)
            if layer_match is None:
                raise ValueError(f"Cannot parse layer index from {layer}")
            ly = int(layer_match.group(1))
            h[0][t] = torch.tensor(new_mlp_up)[ly]
        return h

    # With the patching rules defined, run the patched model in inference.
    with torch.no_grad(), nethook.TraceDict(
        model,
        list(patch_spec.keys()),
        edit_output=patch_rep,
    ) as td:

        target_ids = mt.tokenizer(' '+answer + '</s>', return_tensors='pt')['input_ids'].to('cuda')

        outputs = model(**inp)
        logits = outputs.logits
        ans = torch.argmax(logits, dim=-1)[:, -target_ids.size(1):].squeeze()
        ans_idss = ans.detach().cpu().numpy().tolist()

        textual_ans = mt.tokenizer.decode(ans_idss, skip_special_tokens=True)
        return textual_ans.strip()


def get_transformation_matrixs(langs, logger, model_name):
    for lang_id in range(1,len(langs)):
        lang = langs[lang_id]
        logger.info(f"lang: {lang}")
        train_mlp_acts_en,train_mlp_acts_zh,valid_mlp_acts_en,valid_mlp_acts_zh = [],[],[],[]
    
        with open(f"./data/ncwm/en-{lang}/train.en", encoding="utf-8") as f:
            en_data = f.readlines()
        with open(f"./data/ncwm/en-{lang}/train.{lang}", encoding="utf-8") as g:
            zh_data = g.readlines()
        ind = 0    
        while ind < 500 and ind < len(en_data):
            if ind % 2 == 0: 
                sent = zh_data[ind]
            else:
                sent = zh_data[ind] + ' ' + en_data[ind]
            encodings = mt.tokenizer(sent, return_tensors='pt')
            input_ids = encodings['input_ids'].to('cuda')
            if len(input_ids[0]) > 500:
                ind += 1
                continue
            attention_mask = encodings['attention_mask'].to('cuda')

            att_val,mlp_val,mlp_up,mlp_down,att_post = get_out_mean(mt.model,input_ids,mt.model.device,-1, LAYERS)
            mlp_val = np.array(mlp_val)

            train_mlp_acts_zh.append(mlp_val[:,:])
            
            if ind % 2 == 0: 
                sent = en_data[ind]
            else:
                sent = en_data[ind] + ' ' + en_data[ind]
            encodings = mt.tokenizer(sent, return_tensors='pt')
            input_ids = encodings['input_ids'].to('cuda')
            attention_mask = encodings['attention_mask'].to('cuda')

            att_val,mlp_val,mlp_up,mlp_down,att_post = get_out_mean(mt.model,input_ids,mt.model.device,-1, LAYERS)
            mlp_val = np.array(mlp_val)

            train_mlp_acts_en.append(mlp_val[:,:])
            ind += 1

        train_mlp_acts_zh, train_mlp_acts_en = np.array(train_mlp_acts_zh),np.array(train_mlp_acts_en)    
        transformation_matrixs = []    
        for i in range(LAYERS):
            transformation_matrix = np.linalg.lstsq(train_mlp_acts_zh[:,i,:], train_mlp_acts_en[:,i,:], rcond=None)[0]    
            transformation_matrixs.append(transformation_matrix)

        os.makedirs(f"cache/{model_name}", exist_ok=True)
        with open(f"cache/{model_name}/transformation_matrixs_{lang}.pkl", "wb") as f:
            pickle.dump(transformation_matrixs, f)
        logger.info(f"transformation_matrixs_{lang} saved")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--use_separate", action="store_const", const=True, default=False)
    parser.add_argument("--use_mean", action="store_const", const=True, default=False)
    parser.add_argument("--zero_shot_tuning", action="store_true", default=False,
                        help="Do not apply learned MLP direction; patch with mlp_val only (identity).")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--task", type=str, default="xcopa")
    return parser.parse_args()


# ================================= 函数在上边 ================================

args = args_parser()
util_globals.ZERO_SHOT_TUNING = args.zero_shot_tuning
time_str = time.strftime("%Y%m%d_%H%M%S")
model_file_name = args.model_name
model_file_name = model_file_name.replace("/", "_")
log_suffix = os.environ.get("INCLINE_LOG_SUFFIX", "")
if log_suffix:
    log_suffix = "_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", log_suffix)
log_dir = f"./logs/{time_str}_{model_file_name}{log_suffix}_{os.getpid()}"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, filename=f"{log_dir}/log.txt", format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


model_name = args.model_name
mt = ModelAndTokenizer(model_name)
LAYERS = mt.num_layers
use_separate = args.use_separate
use_mean = args.use_mean
task_name = args.task
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
logger.info(f"use_separate: {use_separate}, use_mean: {use_mean}, zero_shot_tuning: {args.zero_shot_tuning}, random_seed: {args.random_seed}")
logger.info(f"model_name: {model_name}, layers: {LAYERS}")


langs = ['en', 'et','id','it','sw','ta','th','tr','vi','zh']
seen_langs = ['en', 'ar', 'es', 'fr', 'hi', 'id', 'pt', 'sw', 'ta', 'vi', 'zh']
unseen_langs = ['de', 'el', 'et', 'it', 'ja', 'nl', 'ru', 'th', 'tr', 'uk', 'ur']
langs = seen_langs + unseen_langs
# get_transformation_matrixs(langs, logger, model_file_name)

if ';' in task_name:
    task_names = task_name.split(';')
else:
    task_names = [task_name]

k_list = [1]

if 'xcopa' in task_names:
    langs = ['en', 'et','id','it','sw','ta','th','tr','vi','zh']
    # run_xcopa(mt, langs, [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], logger, use_separate, use_mean)
    run_xcopa(mt, langs, k_list, logger, use_separate, use_mean)
    logger.info(f"xcopa done")

if 'xstorycloze' in task_names:
    langs = ['en', 'ar', 'es', 'hi', 'id', 'ru', 'sw', 'zh']
    # run_xstorycloze(mt, langs, [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], logger, use_separate, use_mean)
    run_xstorycloze(mt, langs, k_list, logger, use_separate, use_mean)
    logger.info(f"xstorycloze done")

if 'xwinograd' in task_names:
    langs = ['en', 'fr', 'ja', 'pt', 'ru', 'zh']
    # run_xwinograd(mt, langs, [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], logger, use_separate, use_mean)
    run_xwinograd(mt, langs, k_list, logger, use_separate, use_mean)
    logger.info(f"xwinograd done")

if 'xcsr' in task_names:
    langs = ['en', 'ar', 'de', 'es', 'fr', 'hi', 'it', 'ja', 'nl','pt', 'ru', 'sw', 'ur','vi', 'zh']
    # run_xcsr(mt, langs, [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], logger, use_separate, use_mean)
    run_xcsr(mt, langs, k_list, logger, use_separate, use_mean)
    logger.info(f"xcsr done")

if 'xnli' in task_names:
    langs = ['en', 'ar', 'de', 'el', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
    # run_xnli(mt, langs, [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], logger, use_separate, use_mean)
    run_xnli(mt, langs, k_list, logger, use_separate, use_mean)
    logger.info(f"xnli done")

if 'mzsre' in task_names:
    langs = ['en', 'de', 'es', 'fr', 'pt', 'ru', 'th', 'tr', 'vi', 'zh']
    # run_mzsre(mt, langs, [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], logger, use_separate, use_mean)
    run_mzsre(mt, langs, k_list, logger, use_separate, use_mean)
    logger.info(f"mzsre done")

if 'mgsm' in task_names:
    langs = ['en', 'de', 'es', 'fr', 'ja', 'ru', 'sw', 'th', 'zh']
    run_mgsm(mt, langs, k_list, logger, use_separate, use_mean)
    logger.info(f"mgsm done")

if 'wmt' in task_names:
    langs = ['en', 'de', 'ja', 'ru', 'uk', 'zh']
    # run_wmt(mt, langs, [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], logger, use_separate, use_mean)
    run_wmt(mt, langs, k_list, logger, use_separate, use_mean)
    logger.info(f"wmt done")

if 'flores' in task_names:
    langs = ['en', 'ar', 'el', 'es', 'fr', 'hi', 'ru', 'tr', 'vi', 'zh']
    # run_flores(mt, langs, [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], logger, use_separate, use_mean)
    run_flores(mt, langs, k_list, logger, use_separate, use_mean)
    logger.info(f"flores done")
