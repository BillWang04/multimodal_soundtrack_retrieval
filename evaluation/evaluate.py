import pickle
import torch
from torch import nn
import torch.nn.functional as F
import json
from tqdm import tqdm
import statistics
import numpy as np
import pickle
import os
experiment_data = 'Movie'
experiment_model = 'TaPlo' #All, Im, Plo, Ta, ImPlo, ImTa, TaPlo, Baseline


audio_list = pickle.load(open("./data/meta/movie_all_audio_paths.pkl", 'rb')) # 
fid_to_meta = json.load(open("./data/meta/meta.json"))
with open(f'./evaluation/thetas/{experiment_model}/{experiment_data}/eval_fids.pkl', 'rb') as file:
    eval_fids = pickle.load(file)



def calculate_recall_at_k(ranks, k):
    """Calculate recall@k from a list of ranks"""
    total_relevant = len(ranks)
    found_in_top_k = sum(1 for rank in ranks if rank <= k)
    return found_in_top_k / total_relevant if total_relevant > 0 else 0

def get_rank(score_list, target_fid, audio_list, is_similarity=True):
    """
    Unified ranking function for both similarity and distance
    is_similarity: True for cosine similarity (higher = better), False for distance (lower = better)
    """
    combined_tensor = torch.tensor(score_list)
    sorted_indices = torch.argsort(combined_tensor, descending=is_similarity).tolist()
    
    target_ranks = []
    for idx, audio_path in enumerate(audio_list):
        audio_fid = audio_path.split("/")[-4] + audio_path.split("/")[-3] + audio_path.split("/")[-2]
        if audio_fid == target_fid:
            rank = sorted_indices.index(idx)
            target_ranks.append(rank + 1)
    
    return target_ranks

def dcg_from_ranks(ranks, k=None):
    """DCG 계산: 랭킹이 낮을수록 relevance가 높다 가정"""
    if k:
        ranks = ranks[:k]
    gains = 1 / np.log2(np.array(ranks) + 1)
    return np.sum(gains)

def ndcg_from_ranks(ranks, k=None):
    """nDCG 계산: 랭크 리스트에서 relevance 추정"""
    # I don't know whether this is a good metric?
    #  since there is no inherent ranking between the songs
    # perhaps reddit stuff will be better way to do this
    ranks = np.array(ranks)
    if k:
        ranks = ranks[:k]

    actual_dcg = dcg_from_ranks(ranks, k)

    # ideal: 가장 좋은 랭크 순으로 정렬 (즉, [1, 2, 3, ..., len])
    ideal_ranks = np.arange(1, len(ranks) + 1)
    ideal_dcg = dcg_from_ranks(ideal_ranks, k)

    return actual_dcg / ideal_dcg if ideal_dcg != 0 else 0.

if experiment_model in ['All_CLUB', 'ImPlo', 'ImTa']:

    theta_c_list = torch.load(f"./evaluation/thetas/{experiment_model}/Movie/theta_c.pt")
    theta_a_list = torch.load(f"./evaluation/thetas/{experiment_model}/Movie/theta_a.pt")

elif experiment_model in ['All']:
    
    theta_c_list = torch.load("./evaluation/thetas/All/theta_c.pt")
    theta_a_list = torch.load("./evaluation/thetas/All/theta_a.pt")

elif experiment_model in ['Ta', 'Plo', 'TaPlo']:
    theta_c_list = torch.load(f"./evaluation/thetas/{experiment_model}/Movie/theta_t.pt")
    theta_a_list = torch.load(f"./evaluation/thetas/{experiment_model}/Movie/theta_a.pt")  # You need to load this too

elif experiment_model == 'Im':
    theta_c_list = torch.load(f"./evaluation/thetas/{experiment_model}/Movie/theta_i.pt")
    theta_a_list = torch.load(f"./evaluation/thetas/{experiment_model}/Movie/theta_a.pt")  # You need to load this too

elif experiment_model == 'Baseline':
    theta_a_list = torch.load("./evaluation/thetas/Baseline/theta_a.pt")
    theta_t_list = torch.load("./evaluation/thetas/Baseline/theta_t.pt")
    theta_i_list = torch.load("./evaluation/thetas/Baseline/theta_i.pt")

inference = {}

print(f"Evaluating {experiment_model} model...")

theta_a_norm = F.normalize(theta_a_list, dim=-1)

if experiment_model == "Baseline":
    print("Using baseline AVCA evaluation with combined embeddings...")
    theta_t_norm = F.normalize(theta_t_list, dim=-1)
    theta_i_norm = F.normalize(theta_i_list, dim=-1)
    
    theta_c_combined = (theta_t_norm + theta_i_norm) / 2
    theta_c_norm = F.normalize(theta_c_combined, dim=-1)
else:
    print(f"Using {experiment_model} evaluation with combined embeddings...")
    theta_c_norm = F.normalize(theta_c_list, dim=-1)

print("Computing similarity matrix...")
similarity_matrix = torch.matmul(theta_c_norm, theta_a_norm.T)

completed_fid_list = []
total_ranks = []

print("Computing ranks...")
for i, fid in tqdm(enumerate(eval_fids)):
    if fid not in completed_fid_list:
        completed_fid_list.append(fid)
        
        score_list = similarity_matrix[i].cpu().numpy().tolist()
        
        score_rank = get_rank(score_list, fid, audio_list, is_similarity=True)
        
        if score_rank is not None:
            total_ranks.extend(score_rank)
            inference[fid] = {"meta": fid_to_meta[fid], "inference": score_rank}
            
            with open('inference.json', 'w') as f:
                json.dump(inference, f)

print(f"\nResults for {experiment_model}:")
print(f"Total ranks: {len(total_ranks)}")
print(f"Mean rank: {np.mean(total_ranks):.2f}")
print(f"Median rank: {np.median(total_ranks):.2f}")
print(f"MRR: {sum([1/rank for rank in total_ranks])/len(total_ranks):.4f}")
print(f"NDCG: {ndcg_from_ranks(total_ranks):.4f}")

recall_1 = calculate_recall_at_k(total_ranks, 1)
recall_5 = calculate_recall_at_k(total_ranks, 5)
recall_10 = calculate_recall_at_k(total_ranks, 10)
recall_20 = calculate_recall_at_k(total_ranks, 20)
recall_50 = calculate_recall_at_k(total_ranks, 50)
recall_100 = calculate_recall_at_k(total_ranks, 100)

print(f"Recall@1: {recall_1:.4f}")
print(f"Recall@5: {recall_5:.4f}")
print(f"Recall@10: {recall_10:.4f}")
print(f"Recall@20: {recall_20:.4f}")
print(f"Recall@50: {recall_50:.4f}")
print(f"Recall@100: {recall_100:.4f}")