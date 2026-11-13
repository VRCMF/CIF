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
from sacrebleu import BLEU


punc = ['.</s>','.\n','.',';','!',',','?','\n','</s>','<pad>']

def eval_bleu(predictions, references):
    """Evaluate BLEU score using sacrebleu
    
    Args:
        predictions: List of predicted translation strings
        references: List of lists, where each inner list contains reference strings
    
    Returns:
        BLEU score (float)
    """
    bleu = BLEU()
    # sacrebleu expects predictions as a list and references as a list of lists
    score = bleu.corpus_score(predictions, references)
    return score.score

def load_data_flores(langs):
    """Load FLORES translation data from devtest directory"""
    # Language code to language name mapping (2-letter to language name)
    lang_name_map = {
        'ar': 'Arabic',
        'el': 'Greek',
        'es': 'Spanish',
        'fr': 'French',
        'hi': 'Hindi',
        'ru': 'Russian',
        'tr': 'Turkish',
        'vi': 'Vietnamese',
        'zh': 'Chinese',
    }
    
    # Map 2-letter language codes to 3-letter FLORES codes
    lang_code_map = {
    'en': 'eng',
    'ar': 'ara',
    'el': 'ell', 
    'es': 'spa',
    'fr': 'fra',
    'hi': 'hin',
    'ru': 'rus',
    'tr': 'tur',
    'vi': 'vie',
    'zh': 'zho_simpl',  # 或者用 'cmn' (见下文解释)
}
    
    # Load English reference translations from devtest directory
    english_sentences = []
    en_file = f"data/flores/flores101_dataset/devtest/eng.devtest"
    if os.path.exists(en_file):
        with open(en_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            english_sentences.append(line.strip())
    
    question_all, answer_all = [], []
    
    for lang in langs:
        if lang == 'en':
            # Skip English, it's the target language
            question_all.append([])
            answer_all.append([])
            continue
            
        questions, answers = [], []
        lang_name = lang_name_map.get(lang, lang.capitalize())
        
        # Map 2-letter code to 3-letter FLORES code
        flores_lang_code = lang_code_map.get(lang, lang)
        data_file = f"data/flores/flores101_dataset/devtest/{flores_lang_code}.devtest"
        
        if not os.path.exists(data_file):
            print(f"Warning: Data file not found: {data_file}")
            question_all.append([])
            answer_all.append([])
            continue
            
        with open(data_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for ind, line in enumerate(lines):
            source_text = line.strip()  # Source language text
            
            # Get corresponding English translation
            if ind < len(english_sentences):
                target_text = english_sentences[ind]  # English translation (target)
            else:
                target_text = ""  # Skip if no corresponding English sentence
            
            # Format: Translate the following sentence from {language} to English: {input}
            prompt_question = f"Translate the following sentence from {lang_name} to English: {source_text}"
            
            questions.append(prompt_question)
            answers.append(target_text)
        
        question_all.append(questions)
        answer_all.append(answers)

    return question_all, answer_all



def run_flores(mt, langs, k_list, logger, use_separate, use_mean):
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


    question_all,answer_all = load_data_flores(langs)
    for k in k_list:
        get_results_translation_flores(langs,question_all,answer_all,transformation_matrixs_fused_list, mt, k, logger, use_separate, use_mean)


def get_results_translation_flores(langs,question_all,answer_all,transformation_matrixs_fused_list, mt, k, logger, use_separate, use_mean):

    logger.info(f"k: {k}")
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

                r = trace_with_patch_discrimination(
                        mt.model,
                        inp_question_answer,
                        [(ntoks-1, layername(mt.model, layer,'mlp')) for layer in range(LAYERS)],
                        new_mlp_up = mlp_dires,
                        question=question,
                        answer=answer,
                        mt=mt)
                ans_res.append(r)


            # Evaluate using BLEU score
            references = [[answer_all[lang_id][i]] for i in range(len(ans_res))]  # List of lists for sacrebleu
            bleu_score = eval_bleu(ans_res, references)
            logger.info(f"lang: {langs[lang_id]}, sigma: {sigma}, BLEU: {bleu_score:.2f}")