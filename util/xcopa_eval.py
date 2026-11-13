import os, re, json, pickle
import torch
import torch.nn.functional as F
import json
import numpy as np

from collections import defaultdict
from util import nethook
import util.globals as util_globals
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
from util.base_function import compute_language_agnostic_via_svd_lowrank, compute_language_agnostic_via_concat_svd, trace_with_patch_discrimination, compute_mean_transformation_matrix, compute_language_agnostic_via_svd_2, get_cache_path
from utils import get_out

punc = ['.</s>','.\n','.',';','!',',','?','\n','</s>','<pad>']

def obtain_f1_and_em(mt, a,b):
    global tokenizer
    
    a_words = mt.tokenizer.encode(a, add_special_tokens=False)
    b_words = mt.tokenizer.encode(b, add_special_tokens=False)
    if len(a_words) == 0 and len(b_words) == 0:
        return 1.0, 1
    if len(a_words) == 0 or len(b_words) == 0:
        return 0.0, 0

    em = 1 if a == b else 0
    k = len(a_words) * len(b_words)

    intersecting_words = []
    for word in a_words.copy():
        if word in b_words:
            a_words.remove(word)
            b_words.remove(word)
            intersecting_words.append(word)

    f1_score = (len(intersecting_words) * len(intersecting_words)) / float(k)
    return f1_score, em

def eval_discrimination(mt, ans_res,reference,lg):
    F1, EM = [],[]
    for i in range(len(ans_res)):
        ans = ans_res[i].strip().lower()
        for p in punc:
            ans = ans.replace(p, "")
        f1,em = obtain_f1_and_em(mt, ans,reference[lg][i].strip().lower())
        F1.append(f1)
        EM.append(em)
    F1,EM = mean(F1,axis=0),mean(EM,axis=0)
    return EM,F1

def load_data_xcopa(langs):
    questions_en = []
    with open(f"data/xcopa/data/en/test.en.jsonl", "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        line = json.loads(line)
        ques = line['question']
        questions_en.append(ques)
    question_all,answer_all = [],[]
    for lang in langs:
        questions,answers = [],[]
        with open(f"data/xcopa/data/{lang}/test.{lang}.jsonl", "r", encoding="utf-8") as f:
            lines = f.readlines()
        for ind in range(len(lines)):
            # ipdb.set_trace()
            line = lines[ind]
            line = json.loads(line)
            premise,choice1,choice2,label = line['premise'],line['choice1'],line['choice2'],line['label']
            question = questions_en[ind]
            prompt_question = f'Here is a premise: "{premise}". A: "{choice1}" B: "{choice2}" What is the {question}? "A" or "B"?'

            if int(label) == 0:
                prompt_answer = 'A'
            elif int(label) == 1:
                prompt_answer = 'B'
            questions.append(prompt_question)
            answers.append(prompt_answer)
        question_all.append(questions)
        answer_all.append(answers)

    return question_all,answer_all



def run_xcopa(mt, langs, k_list, logger, use_separate, use_mean):
    transformation_matrixs_fused_list = []
    # transformation_matrixs_fused = None

    for lang_id in range(1,len(langs)):
        lang = langs[lang_id]
        logger.info(f"lang: {lang}")
        with open(get_cache_path(mt.model_name, lang), "rb") as f:
            transformation_matrixs = pickle.load(f)
        transformation_matrixs = np.array(transformation_matrixs)

        transformation_matrixs_fused_list.append(torch.from_numpy(transformation_matrixs).float())
                
        if not os.path.exists(f"plots/fused/"):
            os.makedirs(f"plots/fused/", exist_ok=True)


    question_all,answer_all = load_data_xcopa(langs)
    for k in k_list:
        get_results_discrimination_xcopa(langs,question_all,answer_all,transformation_matrixs_fused_list, mt, k, logger, use_separate, use_mean)


def get_results_discrimination_xcopa(langs,question_all,answer_all,transformation_matrixs_fused_list, mt, k, logger, use_separate, use_mean):

    logger.info(f"k: {k}")
    fusion_method = os.environ.get("XCOPA_FUSION_METHOD")
    decoder = os.environ.get("XCOPA_DECODER", "")
    if decoder:
        logger.info(f"XCOPA_DECODER={decoder}")
    sigma_by_lang = None
    skip_unspecified_sigma_langs = os.environ.get("XCOPA_SKIP_UNSPECIFIED_SIGMA_LANGS") in {"1", "true", "True"}
    sigma_by_lang_raw = os.environ.get("XCOPA_SIGMA_BY_LANG_JSON")
    if sigma_by_lang_raw:
        sigma_by_lang = json.loads(sigma_by_lang_raw)
        logger.info(f"using per-language sigma override: {sigma_by_lang}")
    # logger.info(f"adding sign alignment")
    logger.info(f"compute_language_agnostic_via_svd_lowrank")
    if not use_separate:
        if fusion_method:
            from fusion_methods_xcopa import compute_xcopa_fusion
            transformation_matrixs_fused = compute_xcopa_fusion(
                transformation_matrixs_fused_list, fusion_method, k=k, logger=logger
            )
            LAYERS = transformation_matrixs_fused.shape[0]
        elif use_mean:
            transformation_matrixs_fused = compute_mean_transformation_matrix(transformation_matrixs_fused_list)
            LAYERS = transformation_matrixs_fused.shape[0]
        else:
            # transformation_matrixs_fused = compute_language_agnostic_via_svd_lowrank(transformation_matrixs_fused_list, k)
            transformation_matrixs_fused = compute_language_agnostic_via_svd_2(transformation_matrixs_fused_list, k)
            LAYERS = transformation_matrixs_fused.shape[0]
    for lang_id in range(1,len(langs)):
        if use_separate:
            transformation_matrixs_fused = transformation_matrixs_fused_list[lang_id-1]
            LAYERS = transformation_matrixs_fused.shape[0]
        lang = langs[lang_id]
        counts = len(question_all[lang_id])
        if sigma_by_lang and lang in sigma_by_lang:
            sigma_override = sigma_by_lang[lang]
            if isinstance(sigma_override, list):
                sigmas = [float(sigma) for sigma in sigma_override]
            else:
                sigmas = [float(sigma_override)]
        elif skip_unspecified_sigma_langs:
            sigmas = []
        else:
            sigmas = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        for sigma in sigmas:
            ans_res = []
            for ind in range(counts):
                question = question_all[lang_id][ind]
                answer = answer_all[lang_id][ind]

                inputs = mt.tokenizer.encode(question, return_tensors="pt").to("cuda")
                att_val,mlp_val,mlp_up,mlp_down,att_post = get_out(mt.model,inputs,mt.model.device,-1, LAYERS)
                mlp_val = np.array(mlp_val)
                
                if util_globals.ZERO_SHOT_TUNING:
                    mlp_dires = mlp_val
                else:
                    dires = []

                    for i in range(LAYERS):
                        dire = np.dot([mlp_val[i]], transformation_matrixs_fused[i])[0]
                        # ipdb.set_trace()
                        # dire_1 = np.dot([mlp_val[i]], transformation_matrixs_fused[0][i])[0]
                        # dire_2 = np.dot([mlp_val[i]], transformation_matrixs_fused[1][i])[0]
                        # dire = dire_1 + 1*dire_2
                        dires.append(dire)
                    dires = np.array(dires)
                    mlp_dires = mlp_val + sigma * dires


                inp = make_inputs(mt.tokenizer, [question])
                ntoks = inp["input_ids"].shape[1]       
                question_answer = f'{question} {answer}'
                inp_question_answer = make_inputs(mt.tokenizer, [question_answer])

                if decoder == "choice_logits":
                    from xcopa_decoders import trace_with_patch_choice_logits
                    r = trace_with_patch_choice_logits(
                            mt.model,
                            inp,
                            [(ntoks-1, layername(mt.model, layer,'mlp')) for layer in range(LAYERS)],
                            new_mlp_up = mlp_dires,
                            mt=mt)
                else:
                    r = trace_with_patch_discrimination(
                            mt.model,
                            inp_question_answer,
                            [(ntoks-1, layername(mt.model, layer,'mlp')) for layer in range(LAYERS)],
                            new_mlp_up = mlp_dires,
                            question=question,
                            answer=answer,
                            mt=mt)
                    # decoding 有问题
                    if 'A' in r:
                        r = 'A'
                    else:
                        r = 'B'
                
                ans_res.append(r)


            EM,F1 = eval_discrimination(mt, ans_res,answer_all,lang_id)
            # logger.info(langs[lang_id],sigma,EM)
            logger.info(f"lang: {langs[lang_id]}, sigma: {sigma}, EM: {EM}, F1: {F1}")
