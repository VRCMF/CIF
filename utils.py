import json
import pickle
import re
from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch

from causal_trace import layername, make_inputs
from util import nethook
from util.rome.util import TraceDict
import ipdb


def obtain_f1_and_em(a,b):
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

def eval_discrimination(ans_res,reference,lg):
    F1, EM = [],[]
    for i in range(len(ans_res)):
        ans = ans_res[i].strip().lower()
        for p in punc:
            ans = ans.replace(p, "")
        f1,em = obtain_f1_and_em(ans,reference[lg][i].strip().lower())
        F1.append(f1)
        EM.append(em)
    F1,EM = np.mean(F1,axis=0),np.mean(EM,axis=0)
    return EM,F1

def get_out_mean(model, prompt, device,index, LAYERS): 

    model.eval()
    # ATT = [f"transformer.h.{i}.self_attention.att_out" for i in range(LAYERS)]
    _name = model.model_name.lower() if hasattr(model, "model_name") else model.config._name_or_path.lower()
    if "bloomz" in _name:
        # ATT = [f"transformer.h.{i}.self_attention.dense" for i in range(LAYERS)]
        # QKV = [f"transformer.h.{i}.self_attention.query_key_value" for i in range(LAYERS)]
        MLP = [f"transformer.h.{i}.mlp" for i in range(LAYERS)]
        # MLP_up = [f"transformer.h.{i}.mlp.dense_h_to_4h" for i in range(LAYERS)]
        # MLP_act = [f"transformer.h.{i}.mlp.gelu_impl" for i in range(LAYERS)]
        # MLP_down = [f"transformer.h.{i}.mlp.dense_4h_to_h" for i in range(LAYERS)]
        # ATT_post = [f"transformer.h.{i}.post_attention_layernorm" for i in range(LAYERS)]
        # EMB = [f"transformer.word_embeddings_layernorm"]
        # HEADS = [f"transformer.h.{i}.self_attention" for i in range(LAYERS)]
    elif any(k in _name for k in ["llama", "mistral", "qwen"]):
        # ATT = [f"model.layers.{i}.self_attn" for i in range(LAYERS)]
        # ATT_q = [f"model.layers.{i}.self_attn.q_proj" for i in range(LAYERS)]
        # ATT_k = [f"model.layers.{i}.self_attn.k_proj" for i in range(LAYERS)]
        # ATT_v = [f"model.layers.{i}.self_attn.v_proj" for i in range(LAYERS)]
        # ATT_o = [f"model.layers.{i}.self_attn.o_proj" for i in range(LAYERS)]
        # QKV = ATT_q + ATT_k + ATT_v + ATT_o
        MLP = [f"model.layers.{i}.mlp" for i in range(LAYERS)]
        # MLP_gate = [f"model.layers.{i}.mlp.gate_proj" for i in range(LAYERS)]
        # MLP_up = [f"model.layers.{i}.mlp.up_proj" for i in range(LAYERS)]
        # MLP_act = [f"model.layers.{i}.mlp.act_fn" for i in range(LAYERS)]
        # MLP_down = [f"model.layers.{i}.mlp.down_proj" for i in range(LAYERS)]
        # ATT_post = [f"model.layers.{i}.post_attention_layernorm" for i in range(LAYERS)]
        # EMB = ["model.embed_tokens"]
        # HEADS = [f"model.layers.{i}.self_attn" for i in range(LAYERS)]
    else:
        raise ValueError(f"Model {_name} not supported")

    # ipdb.set_trace()
    with torch.no_grad():
        # with TraceDict(model, ATT+MLP+MLP_up+MLP_down+ATT_post+EMB+HEADS+MLP_act) as ret:
        with TraceDict(model, MLP) as ret:
            output = model(prompt, output_hidden_states = True,output_attentions=True)

        # ATT_value = [np.mean(ret[att_value].output[0].detach().cpu().float().numpy(),axis=0) for att_value in ATT]
        MLP_value = [np.mean(ret[mlp_value].output[0].detach().cpu().float().numpy(),axis=0) for mlp_value in MLP]
        # MLP_up_value = [np.mean(ret[mlp_up].output[0].detach().cpu().float().numpy(),axis=0) for mlp_up in MLP_up]
        # MLP_act_value = [np.mean(ret[mlp_act].output[0].detach().cpu().float().numpy(),axis=0) for mlp_act in MLP_act] 
        # MLP_down_value = [np.mean(ret[mlp_down].output[0].detach().cpu().float().numpy(),axis=0) for mlp_down in MLP_down]
        # ATT_post_value = [np.mean(ret[att_post].output[0].detach().cpu().float().numpy(),axis=0) for att_post in ATT_post]
        # HEAD_value = [np.mean(ret[head_value].output.detach().cpu().numpy(),axis=1) for head_value in HEADS] 
        # HEAD_value = [np.mean(ret[head_value].output[0].detach().cpu().float().numpy(),axis=0) for head_value in HEADS] 
        

        # return ATT_value, MLP_value,MLP_up_value,MLP_down_value,ATT_post_value,HEAD_value,MLP_act_value
        # return ATT_value, MLP_value,MLP_up_value,MLP_down_value,ATT_post_value
        return None, MLP_value,None,None,None


def get_out(model, prompt, device,index, LAYERS): 

    model.eval()
    # 之前的版本
    # ATT = [f"transformer.h.{i}.self_attention.att_out" for i in range(LAYERS)] 
    _name = model.model_name.lower() if hasattr(model, "model_name") else model.config._name_or_path.lower()
    # ipdb.set_trace()
    if "bloomz" in _name:
        # ATT = [f"transformer.h.{i}.self_attention.dense" for i in range(LAYERS)]
        # QKV = [f"transformer.h.{i}.self_attention.query_key_value" for i in range(LAYERS)]
        MLP = [f"transformer.h.{i}.mlp" for i in range(LAYERS)]
        # MLP_up = [f"transformer.h.{i}.mlp.dense_h_to_4h" for i in range(LAYERS)]
        # MLP_act = [f"transformer.h.{i}.mlp.gelu_impl" for i in range(LAYERS)]
        # MLP_down = [f"transformer.h.{i}.mlp.dense_4h_to_h" for i in range(LAYERS)]
        # ATT_post = [f"transformer.h.{i}.post_attention_layernorm" for i in range(LAYERS)]
        # EMB = [f"transformer.word_embeddings_layernorm"]
        # HEADS = [f"transformer.h.{i}.self_attention" for i in range(LAYERS)]
    elif any(k in _name for k in ["llama", "mistral", "qwen"]):
        # ATT = [f"model.layers.{i}.self_attn.o_proj" for i in range(LAYERS)]
        MLP = [f"model.layers.{i}.mlp" for i in range(LAYERS)]
        # MLP_gate = [f"model.layers.{i}.mlp.gate_proj" for i in range(LAYERS)]
        # MLP_up = [f"model.layers.{i}.mlp.up_proj" for i in range(LAYERS)]
        # MLP_act = [f"model.layers.{i}.mlp.act_fn" for i in range(LAYERS)]
        # MLP_down = [f"model.layers.{i}.mlp.down_proj" for i in range(LAYERS)]
        # ATT_post = [f"model.layers.{i}.post_attention_layernorm" for i in range(LAYERS)]
        # EMB = ["model.embed_tokens"]
        # HEADS = [f"model.layers.{i}.self_attn" for i in range(LAYERS)]
    else:
        raise ValueError(f"Model {_name} not supported")

    with torch.no_grad():
        # with TraceDict(model, ATT+MLP+MLP_up+MLP_down+ATT_post+EMB+HEADS+MLP_act) as ret:
        with TraceDict(model, MLP) as ret:
            output = model(prompt, output_hidden_states = True,output_attentions=True)

        # ATT_value = [ret[att_value].output[0,index,:].detach().cpu().float().numpy() for att_value in ATT]
        MLP_value = [ret[mlp_value].output[0,index,:].detach().cpu().float().numpy() for mlp_value in MLP]
        # MLP_up_value = [ret[mlp_up].output[0,index,:].detach().cpu().float().numpy() for mlp_up in MLP_up]
        # MLP_act_value = [ret[mlp_act].output[0,index,:].detach().cpu().float().numpy() for mlp_act in MLP_act] 
        # MLP_down_value = [ret[mlp_down].output[0,index,:].detach().cpu().float().numpy() for mlp_down in MLP_down]
        # ATT_post_value = [ret[att_post].output[0,index,:].detach().cpu().float().numpy() for att_post in ATT_post]
        # HEAD_value = [ret[head_value].output[:,index,:].detach().cpu().numpy() for head_value in HEADS] 
        # HEAD_value = [np.mean(ret[head_value].output[0].detach().cpu().float().numpy(),axis=0) for head_value in HEADS] 
        

        # return ATT_value, MLP_value,MLP_up_value,MLP_down_value,ATT_post_value,HEAD_value,MLP_act_value
        # return ATT_value, MLP_value,MLP_up_value,MLP_down_value,ATT_post_value
        return None, MLP_value,None,None,None


def trace_with_patch(
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
            ly = int(layer.split('transformer.h.')[-1].split('.mlp')[0])
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


def load_data_xcopa():
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


def get_transformation_matrixs():
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

            att_val,mlp_val,mlp_up,mlp_down,att_post = get_out_mean(mt.model,input_ids,mt.model.device,-1)
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

        with open(f"cache/transformation_matrixs_{lang}.pkl", "wb") as f:
            pickle.dump(transformation_matrixs, f)
        logger.info(f"transformation_matrixs_{lang} saved")

def rect_mask_update(dW, W_ref, topk=0.4, eps=1e-8):
    """
    dW: (n, m) update matrix for one language (ΔW)
    W_ref: (n, m) original/base weight matrix (W)
    topk: keep top-k% by relative change |ΔW/W|
    """
    rel = np.abs(dW) / (np.abs(W_ref) + eps)          # δ = |ΔW/W|
    flat = rel.reshape(-1)

    k = int(np.ceil(topk * flat.size))
    if k <= 0:
        return np.zeros_like(dW)

    # threshold for top-k%
    thresh = np.partition(flat, -k)[-k]
    mask = rel <= thresh
    return dW * mask

# sigmas = [-1,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

def get_results_discrimination(langs,question_all,answer_all,transformation_matrixs_fused_list, mt, LAYERS):

    # transformation_matrixs_fused = np.mean(transformation_matrixs_fused_list, axis=0)
    # Replace your mean fusion with consensus fusion (keep everything else)
    transformation_matrixs_fused, _ = fuse_projection_matrices_consensus(
        question_all=question_all,
        langs=langs,
        transformation_matrixs_fused_list=transformation_matrixs_fused_list,
        mt=mt,
        LAYERS=LAYERS,
        max_calib_samples=256,
        calib_lang_ids=list(range(1, len(langs))),
        seed=0,
        per_layer=True,
        normalize_outputs=True,
        center_outputs=True,
    )
    
    for lang_id in range(1,len(langs)):
        lang = langs[lang_id]
        counts = len(question_all[lang_id])
        sigmas = [1]
        for sigma in sigmas:
            ans_res = []
            for ind in range(counts):
                question = question_all[lang_id][ind]
                answer = answer_all[lang_id][ind]

                inputs = mt.tokenizer.encode(question, return_tensors="pt").to("cuda")
                att_val,mlp_val,mlp_up,mlp_down,att_post = get_out(mt.model,inputs,mt.model.device,-1, LAYERS)
                mlp_val = np.array(mlp_val)

                dires = []
                for i in range(LAYERS):
                    dire = np.dot([mlp_val[i]], transformation_matrixs_fused[i])[0]
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
                        answer=answer)
                ans_res.append(r)


            EM,F1 = eval_discrimination(ans_res,answer_all,lang_id)
            # logger.info(langs[lang_id],sigma,EM)
            logger.info(f"lang: {langs[lang_id]}, sigma: {sigma}, EM: {EM}, F1: {F1}")
            
# 使用示例
if __name__ == "__main__":
    # 假设有9个投影矩阵，每个形状为 [12, 768, 768]
    layer_num = 12
    hidden_dim = 768
    
    # 创建示例数据（实际使用时替换为您的真实数据）
    matrices = [
        torch.randn(layer_num, hidden_dim, hidden_dim) for _ in range(9)
    ]
    
    print(f"输入矩阵数量: {len(matrices)}")
    print(f"每个矩阵形状: {matrices[0].shape}")
    
    # 方法1: 基础版本
    print("\n使用方法1: 基础GPU SVD融合")
    fused1 = gpu_svd_fusion(matrices, rank_ratio=0.95, alpha=0.5)
    print(f"融合后形状: {fused1.shape}")
    
    # 方法2: 快速版本（推荐）
    print("\n使用方法2: 快速GPU SVD融合")
    fused2 = fast_gpu_svd_fusion(matrices, rank=None, use_einsum=True)
    print(f"融合后形状: {fused2.shape}")
    
    # 方法3: 内存高效版本（适用于大矩阵）
    print("\n使用方法3: 内存高效GPU SVD融合")
    fused3 = memory_efficient_svd_fusion(matrices, layer_batch_size=4)
    print(f"融合后形状: {fused3.shape}")