import pickle

import torch
# from nltk import util
from sentence_transformers import util

TOP_K = 12
fall_back = [["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["pers:org:member_of", "pers:org:employee_of"],
             ["pers:org:employee_of", "pers:org:employee_of"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["pers:org:member_of", "pers:org:employee_of"],
             ["pers:org:employee_of", "pers:org:employee_of"], ["pers:org:founder_of", "pers:org:founder_of"],
             ["pers:org:employee_of", "pers:org:employee_of"], ["pers:title:title", "pers:title:title"],
             ["pers:title:title", "pers:title:title"], ["pers:title:title", "pers:title:title"],
             ["pers:title:title", "pers:title:title"], ["no_relation", "pers:title:title"],
             ["pers:title:title", "pers:title:title"], ["pers:title:title", "pers:title:title"],
             ["pers:title:title", "pers:title:title"], ["pers:title:title", "pers:title:title"],
             ["pers:title:title", "pers:title:title"], ["pers:title:title", "pers:title:title"],
             ["no_relation", "pers:title:title"], ["pers:title:title", "pers:title:title"],
             ["pers:title:title", "pers:title:title"], ["pers:title:title", "pers:title:title"],
             ["no_relation", "org:gpe:operations_in"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["no_relation", "no_relation"],
             ["org:gpe:operations_in", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "org:gpe:operations_in"], ["no_relation", "org:gpe:operations_in"],
             ["no_relation", "org:gpe:operations_in"], ["no_relation", "org:gpe:operations_in"],
             ["no_relation", "no_relation"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["no_relation", "org:gpe:operations_in"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["no_relation", "org:gpe:operations_in"], ["no_relation", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["no_relation", "no_relation"],
             ["no_relation", "org:gpe:operations_in"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["no_relation", "org:gpe:operations_in"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["no_relation", "org:gpe:operations_in"], ["no_relation", "org:gpe:operations_in"],
             ["no_relation", "org:gpe:operations_in"], ["no_relation", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["org:gpe:operations_in", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["no_relation", "org:gpe:operations_in"], ["no_relation", "org:gpe:operations_in"],
             ["no_relation", "org:gpe:operations_in"], ["no_relation", "org:gpe:operations_in"],
             ["no_relation", "org:gpe:operations_in"], ["no_relation", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["no_relation", "org:gpe:operations_in"],
             ["no_relation", "org:gpe:operations_in"], ["no_relation", "no_relation"],
             ["org:gpe:operations_in", "no_relation"], ["no_relation", "org:gpe:operations_in"],
             ["no_relation", "no_relation"], ["org:money:loss_of", "org:money:loss_of"],
             ["no_relation", "org:money:loss_of"], ["no_relation", "org:money:loss_of"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["org:money:loss_of", "org:money:loss_of"], ["org:money:loss_of", "org:money:loss_of"],
             ["org:money:loss_of", "org:money:loss_of"], ["org:money:loss_of", "org:money:loss_of"],
             ["org:money:revenue_of", "no_relation"], ["org:money:revenue_of", "no_relation"],
             ["org:money:revenue_of", "no_relation"], ["org:money:revenue_of", "no_relation"],
             ["org:money:loss_of", "org:money:loss_of"], ["org:money:loss_of", "no_relation"],
             ["org:money:profit_of", "no_relation"], ["org:money:revenue_of", "org:money:revenue_of"],
             ["no_relation", "org:money:revenue_of"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["org:money:loss_of", "org:money:loss_of"], ["org:money:revenue_of", "org:money:revenue_of"],
             ["org:money:revenue_of", "org:money:revenue_of"], ["org:money:revenue_of", "org:money:revenue_of"],
             ["org:money:revenue_of", "org:money:revenue_of"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["org:money:loss_of", "org:money:loss_of"],
             ["no_relation", "org:money:loss_of"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["org:money:revenue_of", "org:money:revenue_of"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["org:date:formed_on", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["org:date:formed_on", "org:date:formed_on"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["org:date:formed_on", "org:date:formed_on"], ["org:date:formed_on", "org:date:formed_on"],
             ["no_relation", "org:date:formed_on"], ["org:date:formed_on", "org:date:formed_on"],
             ["no_relation", "org:date:formed_on"], ["no_relation", "no_relation"],
             ["org:date:formed_on", "org:date:formed_on"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["org:date:formed_on", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["org:org:agreement_with", "org:org:agreement_with"], ["org:org:agreement_with", "no_relation"],
             ["org:date:formed_on", "no_relation"], ["org:date:formed_on", "org:date:formed_on"],
             ["org:date:formed_on", "no_relation"], ["org:date:formed_on", "org:date:formed_on"],
             ["org:org:subsidiary_of", "no_relation"], ["org:org:shares_of", "no_relation"],
             ["org:org:shares_of", "no_relation"], ["org:money:loss_of", "no_relation"],
             ["org:money:loss_of", "org:money:loss_of"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "org:money:revenue_of"], ["no_relation", "no_relation"],
             ["no_relation", "org:money:loss_of"], ["no_relation", "org:gpe:operations_in"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "pers:org:employee_of"],
             ["no_relation", "no_relation"], ["no_relation", "org:money:revenue_of"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "org:gpe:operations_in"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "org:gpe:operations_in"],
             ["no_relation", "no_relation"], ["no_relation", "pers:title:title"], ["no_relation", "no_relation"],
             ["no_relation", "org:money:loss_of"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "pers:org:employee_of"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "org:date:formed_on"], ["no_relation", "no_relation"],
             ["no_relation", "org:date:formed_on"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "org:money:revenue_of"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "no_relation"], ["no_relation", "pers:org:employee_of"], ["no_relation", "no_relation"],
             ["no_relation", "pers:org:employee_of"], ["no_relation", "no_relation"], ["no_relation", "no_relation"],
             ["no_relation", "org:money:loss_of"], ["no_relation", "no_relation"],
             ["no_relation", "org:date:formed_on"], ["pers:title:title", "pers:title:title"],
             ["pers:title:title", "pers:title:title"], ["pers:title:title", "pers:title:title"],
             ["pers:title:title", "pers:title:title"], ["pers:title:title", "pers:title:title"],
             ["pers:title:title", "pers:title:title"], ["pers:title:title", "pers:title:title"],
             ["pers:title:title", "pers:title:title"], ["pers:title:title", "pers:title:title"],
             ["pers:title:title", "pers:title:title"], ["pers:title:title", "no_relation"],
             ["pers:title:title", "pers:title:title"], ["pers:title:title", "pers:title:title"],
             ["pers:title:title", "pers:title:title"], ["pers:title:title", "no_relation"],
             ["pers:title:title", "no_relation"], ["pers:title:title", "pers:title:title"],
             ["pers:title:title", "no_relation"], ["pers:title:title", "pers:title:title"],
             ["pers:title:title", "no_relation"], ["pers:title:title", "pers:title:title"],
             ["pers:title:title", "pers:title:title"], ["pers:title:title", "pers:title:title"],
             ["pers:title:title", "pers:title:title"], ["pers:title:title", "pers:title:title"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["org:gpe:operations_in", "no_relation"],
             ["org:gpe:operations_in", "no_relation"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "no_relation"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "no_relation"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "no_relation"], ["org:gpe:operations_in", "org:gpe:operations_in"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["org:gpe:operations_in", "no_relation"],
             ["org:gpe:operations_in", "org:gpe:operations_in"], ["pers:org:employee_of", "pers:org:employee_of"],
             ["pers:org:employee_of", "pers:org:employee_of"], ["pers:org:employee_of", "pers:org:employee_of"],
             ["pers:org:employee_of", "pers:org:employee_of"], ["pers:org:employee_of", "pers:org:employee_of"],
             ["pers:org:employee_of", "pers:org:employee_of"], ["pers:org:employee_of", "pers:org:employee_of"],
             ["pers:org:employee_of", "pers:org:employee_of"], ["pers:org:employee_of", "pers:org:employee_of"],
             ["pers:org:employee_of", "pers:org:employee_of"], ["pers:org:employee_of", "pers:org:employee_of"],
             ["pers:org:employee_of", "pers:org:employee_of"], ["pers:org:employee_of", "pers:org:employee_of"],
             ["pers:org:employee_of", "pers:org:employee_of"], ["pers:org:employee_of", "pers:org:employee_of"],
             ["pers:org:employee_of", "pers:org:employee_of"], ["pers:org:employee_of", "pers:org:employee_of"],
             ["org:org:agreement_with", "org:org:agreement_with"], ["org:org:agreement_with", "no_relation"],
             ["org:date:formed_on", "org:date:formed_on"], ["org:date:formed_on", "org:date:formed_on"],
             ["org:date:formed_on", "no_relation"], ["pers:org:member_of", "pers:org:employee_of"],
             ["org:org:shares_of", "no_relation"], ["org:org:shares_of", "no_relation"],
             ["org:money:revenue_of", "no_relation"], ["org:money:revenue_of", "org:money:revenue_of"],
             ["org:money:loss_of", "org:money:loss_of"], ["org:money:loss_of", "org:money:loss_of"],
             ["org:money:loss_of", "no_relation"], ["org:money:loss_of", "org:money:loss_of"],
             ["org:money:loss_of", "org:money:loss_of"], ["org:money:loss_of", "org:money:loss_of"],
             ["org:money:loss_of", "org:money:loss_of"], ["org:money:loss_of", "org:money:loss_of"],
             ["org:money:loss_of", "no_relation"], ["org:date:acquired_on", "no_relation"],
             ["pers:org:founder_of", "pers:org:employee_of"], ["pers:org:founder_of", "pers:org:employee_of"],
             ["pers:univ:member_of", "pers:univ:member_of"], ["no_relation", "org:money:revenue_of"],
             ["no_relation", "no_relation"], ["no_relation", "no_relation"], ["no_relation", "org:money:loss_of"]]
with open('sdp_dr_ner_train_embedding.pickle', 'rb') as handle:
    all_embed = pickle.load(handle)

with open('sdp_dr_ner_dev_embedding.pickle', 'rb') as handle:
    query_dic = pickle.load(handle)

from statistics import mean
from sklearn.metrics import f1_score

y_true = []
y_pred = []

fail_count = 0
avg_sc = 0
lin_c = 0
for qt in query_dic:
    lin_c = lin_c + 1

    # print(qt)
    # query_list.append([query_embedding,data["relation"],data["rel_group"]])
    small_dic = {}
    embed_dic = all_embed[qt[2]]
    # for k in query_dic[RC]:
    query_embedding = qt[0]
    real_class = qt[1]

    max_clas = ""
    max_score = 0
    for ki, vi in embed_dic.items():
        # print("top for ", ki)
        cos_scores = util.cos_sim(query_embedding, vi)[0]
        top_results = torch.topk(cos_scores, k=TOP_K)

        # print("Query:", res)
        sc = (mean(top_results[0].tolist()))
        if sc > max_score:
            max_score = sc
            max_clas = ki
    y_true.append(real_class)
    y_pred.append(max_clas.strip())


for ft in fall_back:
    y_pred.append(ft[1])
    y_true.append(ft[0])

print(f1_score(y_true, y_pred, average='micro'))
