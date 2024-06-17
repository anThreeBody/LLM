import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
from utils import fix_seed

# 在推理之前聚类,将语义上相似的数据聚类

def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")
    parser.add_argument(
        "--task", type=str, default="hotpot",
        choices=["hotpot","aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq",
                 "coin_flip", "last_letters"], help="dataset used for experiment"
    )
    parser.add_argument(
        "--pred_file", type=str, default="./new-dataset/musique\processed_test.json",
        help="use the reasoning chains generated by zero-shot-cot."
    )
    parser.add_argument(
        "--max_ra_len", type=int, default=5, help="maximum number of reasoning chains"
    )
    parser.add_argument(
        "--demo_save_dir", type=str, default="demos/multiarith", help="where to save the contructed demonstrations"
    )
    parser.add_argument("--random_seed", type=int, default=192, help="random seed")
    parser.add_argument(
        "--encoder", type=str, default="all-MiniLM-L6-v2", help="which sentence-transformer encoder for clustering"
    )
    parser.add_argument(
        "--sampling", type=str, default="center", help="whether to sample the cluster center first"
    )
    parser.add_argument(
        "--debug", type=bool, default=True, help="debug mode"
    )
    args = parser.parse_args()
    return args

def cluster_and_split_dataset(train_data_path,test_data_path):
    args = parse_arguments()
    fix_seed(args.random_seed)
    encoder = SentenceTransformer(args.encoder)

    task = args.task
    pred_file = args.pred_file
    save_file = args.demo_save_dir
    max_ra_len = args.max_ra_len
    if task == "last_letters":
        max_ra_len = 7
    if task == "aqua" or task == "last_letters":
        num_clusters = 4
    elif task == "commonsensqa":
        num_clusters = 7
    elif task == "strategyqa":
        num_clusters = 6
    else:
        num_clusters = 8

    corpus = []
    question = []
    rationale = []
    gold_ans = []
    pred_ans = []

    data = []
    with open (pred_file, "r", encoding="utf-8") as f:
        data=json.load(f)
    
    for i in range(len(data)):
        c_question = data[i]['question']
        c_gold_ans = data[i]['answer']
        #c_rationale =  data[i]['evidence']
        corpus.append(c_question)
        question.append(c_question)
        #rationale.append(c_rationale)
        gold_ans.append(c_gold_ans)

    # 利用sentence-transformer对corpus进行编码
    corpus_embeddings = encoder.encode(corpus)
    #print(corpus_embeddings)
    # Perform kmean clustering kmeans聚类
    clustering_model = KMeans(n_clusters=num_clusters, random_state=args.random_seed)
    clustering_model.fit(corpus_embeddings)

    # 获取聚类结果 []list，里面是每个句子的聚类结果，0,1这种聚类的索引
    cluster_assignment = clustering_model.labels_
    # print(cluster_assignment)
    # 将聚类结果按照类别进行分组 [[], [], [], [], [], [], [], []]
    clustered_sentences = [[] for i in range(num_clusters)]
    # print(clustered_sentences)
    # 获取每个簇中每个句子到聚类中心的距离 [[]]
    dist = clustering_model.transform(corpus_embeddings)
    # print(dist)
    # 获取每个句子在各自簇中的索引[]
    clustered_dists = [[] for i in range(num_clusters)]
    # print(clustered_dists)
    # 获取每个句子在各自簇中的索引
    clustered_idx = [[] for i in range(num_clusters)]
    # print(clustered_idx)
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])
        clustered_dists[cluster_id].append(dist[sentence_id][cluster_id])
        clustered_idx[cluster_id].append(sentence_id)

    # 将一半数据作为训练集，一半数据作为测试集，并保存在json文件中
    with open(train_data_path, "w", encoding="utf-8") as train_file:
        with open(test_data_path, "w", encoding="utf-8") as test_file:
            train_data = []
            test_data = []
            # 遍历每个簇，将其中一半作为训练集，一半作为测试集
            for i in range((len(clustered_dists))):
                tmp_train = {}
                tmp_test = {}
                tmp_train["cluster_id"] = i
                tmp_test["cluster_id"] = i
                tmp_array_train = []
                tmp_array_test = []
                # 取前一半作为训练集
                for j in range(len(clustered_sentences[i]) // 2):
                    temp = {}
                    temp["index"] = clustered_idx[i][j]
                    temp["question"] = question[clustered_idx[i][j]]
                    #temp["rationale"] = ""
                    temp["pred_ans"] = ""
                    temp["gold_ans"] = gold_ans[clustered_idx[i][j]]
                    temp["cluster_dist"] = (float(clustered_dists[i][j]))
                    tmp_array_train.append(temp)
                tmp_train["train_samples"] = tmp_array_train
                train_data.append(tmp_train)
                # 后一半作为测试集
                for k in range(len(clustered_sentences[i]) // 2, len(clustered_sentences[i])):
                    temp = {}
                    temp["index"] = clustered_idx[i][k]
                    temp["question"] = question[clustered_idx[i][k]]
                    #temp["rationale"] = ""
                    temp["pred_ans"] = ""
                    temp["gold_ans"] = gold_ans[clustered_idx[i][k]]
                    temp["cluster_dist"] = (float(clustered_dists[i][k]))
                    tmp_array_test.append(temp)
                tmp_test["test_samples"] = tmp_array_test
                test_data.append(tmp_test)
            json.dump(train_data, train_file, ensure_ascii=False, indent=4)
            json.dump(test_data, test_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    cluster_and_split_dataset("./dataset/train/musique_train.json","./dataset/test/musique_test.json")