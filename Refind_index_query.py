import copy
import json
import pickle

import torch
from sentence_transformers import SentenceTransformer, util

import networkx as nx
import spacy
from sentence_transformers import SentenceTransformer

REL_CHECK = ["ORG-ORG", "PERSON-UNIV", "PERSON-GOV_AGY", "PERSON-ORG", "PERSON-TITLE", "ORG-GPE", "ORG-MONEY",
             "ORG-DATE"]
TOP_K = 7
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

nlp = spacy.load('en_core_web_sm')


def parse_str(sent, e1, e2, e1_n, e2_n):
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

# with open('sdp_dr_ner_train_embedding.pickle', 'rb') as handle:
#     all_embed = pickle.load(handle)

# with open('sdp_dr_dev_embedding.pickle', 'rb') as handle:
#     relation_dic = pickle.load(handle)

# for RC in REL_CHECK:
orig_file2 = "input_data/dev_refind_clean.json"
file_inp = open(orig_file2, "r")
from statistics import mean

import numpy as np
from sklearn.metrics import f1_score

y_true = []
y_pred = []

fail_count = 0
line_count = 0
avg_sc = 0
query_dic = {}
# for RC in REL_CHECK:
#     small_dic = {}
query_list = []
for k in file_inp.readlines():
    line_count = line_count+1
    data = json.loads(k)
    li2 = copy.deepcopy(data)
    # if RC not in data['rel_group']:
    #     continue
    del li2['sentence']
    del li2['relation']
    del li2['rel_group']
    ets = list(li2.values())
    et_keys = list(li2.keys())
    try:
        res = parse_str(data['sentence'], ets[0].lower(), ets[1].lower(), et_keys[0].lower(), et_keys[1].lower())
    except:
        fail_count = fail_count + 1
        y_pred.append("no_relation")
        y_true.append(data["relation"].strip())
        print("FAIL\t"+ str(line_count)+"\t"+data["relation"].strip())
        continue
    query_embedding = model.encode(res, convert_to_tensor=True)
    query_list.append([query_embedding,data["relation"],data["rel_group"]])
    # query_dic[RC] = small_dic
with open('sdp_ner_dev_embedding.pickle', 'wb') as handle:
        pickle.dump(query_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

