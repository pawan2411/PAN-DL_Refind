import copy
import json
import pickle

import networkx as nx
import spacy
import torch
from sentence_transformers import SentenceTransformer, util


REL_CHECK = ["ORG-ORG", "PERSON-UNIV", "PERSON-GOV_AGY", "PERSON-ORG", "PERSON-TITLE", "ORG-GPE", "ORG-MONEY",
             "ORG-DATE"]
TOP_K = 7
model = SentenceTransformer('D:/projects/mp-net/')

nlp = spacy.load('en_core_web_sm')


# https://spacy.io/docs/usage/processing-text
def parse_str(sent, e1, e2, e1_n, e2_n):
    # return sent
    document = nlp(sent)
    tok_1 = ""
    tok_2 = ""
    edges = []
    pos_dic = {}
    ner_dic = {}
    for word in document.ents:
        ner_dic[word.text.lower()] = word.label_

    for token in document:
        pos_dic[token.lower_] = token.dep_
        for child in token.children:
            edges.append(('{0}-{1}'.format(token.lower_, token.i),
                          '{0}-{1}'.format(child.lower_, child.i)))
        if not token.lower_.isalpha():
            continue
        if token.lower_ in e1:
            tok_1 = token.lower_ + "-" + str(token.i)
        if token.lower_ in e2:
            tok_2 = token.lower_ + "-" + str(token.i)

    graph = nx.Graph(edges)
    res = (nx.shortest_path(graph, source=tok_1, target=tok_2))
    res_str = e1_n.split("_")[0]
    # + "/" + pos_dic[e1.split(" ")[0]]
    # res_str = e1+" (e1/"+e1_n+")"
  #SDP
    #for k in res[1:-1]:
     #   res_str = res_str + " " + k.split("-")[0]# + "/" + pos_dic[k.split("-")[0]]
    #return res_str + " " + e2_n.split("_")[0]
  #SDP+NER
    #for k in res[1:-1]:
     #   res_str = res_str + " " + k.split("-")[0]# 
      #  if k.split("-")[0] in ner_dic:
       #     res_str = res_str + "/" + ner_dic[k.split("-")[0]]
    #return res_str + " " + e2_n.split("_")[0]
  #SDP+DEP
    #for k in res[1:-1]:
     #   res_str = res_str + " " + k.split("-")[0] + "/" + pos_dic[k.split("-")[0]]
    #return res_str + " " + e2_n.split("_")[0]
  #SDP+DEP+NER
    for k in res[1:-1]:
        res_str = res_str + " " + k.split("-")[0] + "/" + pos_dic[k.split("-")[0]]
        if k.split("-")[0] in ner_dic:
            res_str = res_str + "/" + ner_dic[k.split("-")[0]]
    return res_str + " " + e2_n.split("_")[0]


global_res_dic = {}
for RC in REL_CHECK:
    # print(parse_str("John killed Mary", "john", "mary"))
    orig_file = "C:/Users/Admin/Downloads/train_refind_clean.json"
    file_inp = open(orig_file, "r")

    out_file = open("sample_dep.txt", "w")
    fail_count = 0
    res_dic = {}
    for k in file_inp.readlines():
        data = json.loads(k)
        li2 = copy.deepcopy(data)
        if RC not in data['rel_group']:
            continue
        del li2['sentence']
        del li2['relation']
        del li2['rel_group']
        ets = list(li2.values())
        et_keys = list(li2.keys())
        try:
            res = parse_str(data['sentence'], ets[0].lower(), ets[1].lower(), et_keys[0].lower(), et_keys[1].lower())
        except:
            fail_count = fail_count + 1
            # print("FAIL", fail_count)
            continue
        # out_file.write(data['sentence'] + "\n")
        # out_file.write(data['relation'] + "\n")
        # out_file.write(data['rel_group'] + "\n")
        # out_file.write(str(li2) + "\n")+
        # out_file.write(res + "\n")
        # out_file.write("----\n" + "\n")
        if data['relation'] in res_dic:
            res_dic[data['relation']] = \
                res_dic[data['relation']] + [res]
        else:
            res_dic[data['relation']] = [res]

    # sentences = ["This is an example sentence", "Each sentence is converted"]

    # print(embeddings)
    embed_dic = {}
    for k, v in res_dic.items():
        # print(k, len(v))
        embeddings = model.encode(v)
        embed_dic[k] = embeddings
    global_res_dic[RC] = embed_dic
with open('sdp_ner_train_embedding.pickle', 'wb') as handle:
    pickle.dump(global_res_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # orig_file2 = "C:/Users/Admin/Downloads/dev_refind_clean.json"
    # file_inp = open(orig_file2, "r")
    # from statistics import mean
    #
    # import numpy as np
    # from sklearn.metrics import f1_score
    #
    # y_true = []
    # y_pred = []
    #
    # fail_count = 0
    # avg_sc=0
    # for k in file_inp.readlines():
    #     data = json.loads(k)
    #     li2 = copy.deepcopy(data)
    #     if RC not in data['rel_group']:
    #         continue
    #     del li2['sentence']
    #     del li2['relation']
    #     del li2['rel_group']
    #     ets = list(li2.values())
    #     et_keys = list(li2.keys())
    #     try:
    #         res = parse_str(data['sentence'], ets[0].lower(), ets[1].lower(), et_keys[0].lower(), et_keys[1].lower())
    #     except:
    #         fail_count = fail_count + 1
    #         y_pred.append("no_relation")
    #         y_true.append(data["relation"].strip())
    #         print("FAIL ", fail_count,data["relation"].strip())
    #         continue
    #     query_embedding = model.encode(res, convert_to_tensor=True)
    #
    #     # print("Orig: ", data["relation"])
    #     max_score = 0
    #     max_clas = ""
    #     for ki, vi in embed_dic.items():
    #         # print("top for ", ki)
    #         cos_scores = util.cos_sim(query_embedding, vi)[0]
    #         top_results = torch.topk(cos_scores, k=TOP_K)
    #
    #         # print("Query:", res)
    #         sc = (mean(top_results[0].tolist()))
    #         if sc > max_score:
    #             max_score = sc
    #             max_clas = ki
    #
    #         # for score, idx in zip(top_results[0], top_results[1]):
    #         #     print(res_dic[ki][idx], "(Score: {:.4f})".format(score))
    #     print(max_clas.strip(), data["relation"].strip())
