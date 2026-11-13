import os
import os, re, json
import torch
import json
import numpy as np
from collections import defaultdict
from util import nethook
import util.globals as util_globals
from baukit import Trace, TraceDict
import torch.nn.functional as F
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
from transformers import AutoTokenizer,AutoModelForCausalLM,GenerationConfig
from util.base_function import compute_language_agnostic_via_svd_lowrank, compute_language_agnostic_via_concat_svd, compute_mean_transformation_matrix, trace_with_patch_generative, compute_language_agnostic_via_svd_2, get_cache_path
from utils import get_out
torch.set_grad_enabled(False)
import pickle


punc = ['.</s>','.\n','.',';','!',',','?','\n','</s>','<pad>']




def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  
    res = re.findall(r"(\d+(\.\d+)?)", text) 
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0


def eval_generative(ans_res,reference,lg):
    res = []
    for i in range(len(ans_res)):
        ans = ans_res[i].strip().lower()
        for p in punc:
            ans = ans.replace(p, "")
        extract_true_num = extract_last_num(reference[lg][i].strip().lower())
        extract_pred_num = extract_last_num(ans)
        if abs(extract_true_num - extract_pred_num) < 1e-3:
            res.append(1)
        else:
            res.append(0)
    EM = np.mean(res,axis=0)
    # For generative tasks, F1 is the same as EM (exact match)
    F1 = EM
    return EM, F1

def load_data_mgsm(langs=None):
    if langs is None:
        langs = ["en", "de","es", "fr", "ja", "ru", "sw", "th", "zh"]
    names = {'en':'English','de':'German','es':'Spanish','fr':'French','ja':'Japanese','ru':'Russian','sw':'Swahili','th':'Thai','zh':'Chinese'}
    question_all, answer_all = [],[]
    for lang in langs:
        name = names[lang]
        questions,answers = [],[]
        file_path = f"data/mgsm/mgsm_{lang}.jsonl"
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            question_all.append([])
            answer_all.append([])
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                question = data.get('question', '')
                # Use answer_number if answer is None, otherwise use answer
                answer = data.get('answer')
                if answer is None:
                    answer = str(data.get('answer_number', ''))
                else:
                    answer = str(answer)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON line: {line[:100]}... Error: {e}")
                continue
            # prompt_no_input = (
            #       "Below is an instruction that describes a task. "
            #         f"Write a response that appropriately completes the request in {name}. Please answer in {name}.\n\n"
            #         "### Instruction:\n{query}\n\n### Response:"
            #     )
            prompt = f"Write a response that appropriately completes the request in {name}. Please answer in {name}.\n\n ### Instruction: {question}\n\n### Response:"
            # prompt = prompt_no_input.format(query=question)
            questions.append(prompt)
            answers.append(answer)
            
        question_all.append(questions)
        answer_all.append(answers)
        
    print(len(question_all),len(question_all[0]) if question_all and question_all[0] else 0)
    if question_all and question_all[-1]:
        print(question_all[-1][0])
    print(len(answer_all),len(answer_all[0]) if answer_all and answer_all[0] else 0)
    if answer_all and answer_all[-1]:
        print(answer_all[-1][0])      
    return question_all,answer_all



def run_mgsm(mt, langs, k_list, logger, use_separate, use_mean):
    if mt.tokenizer.pad_token is None:
        mt.tokenizer.pad_token = mt.tokenizer.eos_token
    transformation_matrixs_fused_list = []
    # transformation_matrixs_fused = None

    for lang_id in range(1,len(langs)):
        lang = langs[lang_id]
        logger.info(f"lang: {lang}")
        with open(get_cache_path(mt.model_name, lang), "rb") as f:
            transformation_matrixs = pickle.load(f)
        transformation_matrixs = np.array(transformation_matrixs)

        transformation_matrixs_fused_list.append(torch.from_numpy(transformation_matrixs).float())
                
        # if not os.path.exists(f"plots/fused/"):
        #     os.makedirs(f"plots/fused/", exist_ok=True)


    question_all,answer_all = load_data_mgsm(langs)
    for k in k_list:
        get_results_generative_mgsm(langs,question_all,answer_all,transformation_matrixs_fused_list, mt, k, logger, use_separate, use_mean)

def get_results_generative_mgsm(langs,question_all,answer_all,transformation_matrixs_fused_list, mt, k, logger, use_separate, use_mean):

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
        sigmas = [1]
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
                        dire = -np.dot([mlp_val[i]], transformation_matrixs_fused[i])[0]
                        dires.append(dire)
                    dires = np.array(dires)
                    mlp_dires = mlp_val + sigma * dires


                inp = make_inputs(mt.tokenizer, [question])
                ntoks = inp["input_ids"].shape[1]       
                question_answer = f'{question} {answer}'
                inp_question_answer = make_inputs(mt.tokenizer, [question_answer])
                
                r = trace_with_patch_generative(
                        mt.model,
                        inp_question_answer,
                        [(ntoks-1, layername(mt.model, layer,'mlp')) for layer in range(LAYERS)],
                        new_mlp_up = mlp_dires,
                        question=question,
                        answer=answer,
                        mt=mt)
                # ipdb.set_trace()
                ans_res.append(r)


            EM,F1 = eval_generative(ans_res,answer_all,lang_id)
            # logger.info(langs[lang_id],sigma,EM)
            logger.info(f"lang: {langs[lang_id]}, sigma: {sigma}, EM: {EM}, F1: {F1}")
