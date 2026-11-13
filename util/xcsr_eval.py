import os
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

def load_data_xcsr(langs):
    """
    Load xcsr data from local JSONL files.
    Data directory: data/xcsr
    Available languages: en, zh, de, es, fr, it, jap, nl, pl, pt, ru, ar, vi, hi, sw, ur
    Data format: question with stem and choices, answerKey (A, B, C, D, or E)
    """
    question_all,answer_all = [],[]
    
    # Map language codes to xcsr file language codes
    xcsr_lang_map = {
        'en': 'en',
        'zh': 'zh',
        'de': 'de',
        'es': 'es',
        'fr': 'fr',
        'it': 'it',
        'ja': 'jap',
        'nl': 'nl',
        'pl': 'pl',
        'pt': 'pt',
        'ru': 'ru',
        'ar': 'ar',
        'vi': 'vi',
        'hi': 'hi',
        'sw': 'sw',
        'ur': 'ur',
        'jp': 'jap',
    }
    
    data_dir = "data/xcsr"
    
    for lang in langs:
        questions,answers = [],[]
        
        # Map language code to xcsr language code
        xcsr_lang = xcsr_lang_map.get(lang, None)
        
        if xcsr_lang is None:
            # Skip if language not available in xcsr
            question_all.append([])
            answer_all.append([])
            continue
        
        try:
            file_path = os.path.join(data_dir, f"xcsr_{xcsr_lang}.jsonl")
            
            if not os.path.exists(file_path):
                print(f"Warning: File not found for language {lang} ({xcsr_lang}): {file_path}")
                question_all.append([])
                answer_all.append([])
                continue
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    question_stem = item['question']['stem']
                    choices = item['question']['choices']
                    answer_key = item.get('answerKey', None)
                    
                    # Format choices as A, B, C, D, E (matching the format from load_dataset.py)
                    choice_texts = []
                    for choice_idx in range(len(choices['label'])):
                        label = choices['label'][choice_idx]
                        text = choices['text'][choice_idx]
                        # choice_texts.append(f'{text}')
                        choice_texts.append(f'{label}: {text}')
                    
                    choices_str = '\n'.join(choice_texts)
                    prompt_question = f'Question: {question_stem}\n{choices_str}\nAnswer:'
                    
                    # Create prompt: question with choices, ask which is correct
                    # Format similar to xcopa but adapted for 5 choices
                    # prompt_question = f'Question: {question_stem} {choices_str}. Answer:'
                    
                    # Answer is the answerKey (A, B, C, D, or E)
                    # if answer_key is not None:
                    # if answer_key == 'A':
                    #     prompt_answer = choices['text'][0]
                    # elif answer_key == 'B':
                    #     prompt_answer = choices['text'][1]
                    # elif answer_key == 'C':
                    #     prompt_answer = choices['text'][2]
                    # elif answer_key == 'D':
                    #     prompt_answer = choices['text'][3]
                    # elif answer_key == 'E':
                    #     prompt_answer = choices['text'][4]
                    prompt_answer = answer_key
                    # ipdb.set_trace()
                    questions.append(prompt_question)
                    answers.append(prompt_answer)
        
        except Exception as e:
            # If dataset loading fails, skip this language
            print(f"Wrong: Failed to load xcsr data for language {lang} ({xcsr_lang}): {e}")
            question_all.append([])
            answer_all.append([])
            continue
        
        question_all.append(questions)
        answer_all.append(answers)
    
    return question_all,answer_all


def run_xcsr(mt, langs, k_list, logger, use_separate, use_mean):
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


    question_all,answer_all = load_data_xcsr(langs)
    for k in k_list:
        get_results_discrimination_xcsr(langs,question_all,answer_all,transformation_matrixs_fused_list, mt, k, logger, use_separate, use_mean)


def get_results_discrimination_xcsr(langs,question_all,answer_all,transformation_matrixs_fused_list, mt, k, logger, use_separate, use_mean):

    logger.info(f"k: {k}")
    decoder = os.environ.get("INCLINE_DECODER", "")
    if decoder:
        logger.info(f"INCLINE_DECODER={decoder}")
    # logger.info(f"adding sign alignment")
    logger.info(f"compute_language_agnostic_via_svd_lowrank")
    if not use_separate:
        fusion_method = os.environ.get("INCLINE_FUSION_METHOD") or os.environ.get("XCOPA_FUSION_METHOD")
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
        if counts == 0:
            logger.info(f"lang: {langs[lang_id]}, no data available, skipping")
            continue
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

                if decoder == "candidate_logits":
                    from candidate_decoders import trace_with_patch_candidate_logprobs
                    r = trace_with_patch_candidate_logprobs(
                            mt.model,
                            question,
                            ["A", "B", "C", "D", "E"],
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
                    if 'A' in r:
                        r = 'A'
                    elif 'B' in r:
                        r = 'B'
                    elif 'C' in r:
                        r = 'C'
                    elif 'D' in r:
                        r = 'D'
                    elif 'E' in r:
                        r = 'E'
                ans_res.append(r)


            EM,F1 = eval_discrimination(mt, ans_res,answer_all,lang_id)
            # ipdb.set_trace()
            # logger.info(langs[lang_id],sigma,EM)
            logger.info(f"lang: {langs[lang_id]}, sigma: {sigma}, EM: {EM}, F1: {F1}")
