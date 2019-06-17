import os
import time
import numpy

def recall_2at1(score_list, k=1):
    num_correct = 0
    num_total = len(score_list)
    for scores in score_list:
        ranking_index = numpy.argsort(-numpy.array(scores[0:2]))
        # Message at index 0 is always correct next message in our test data
        if 0 in ranking_index[:k]:
            num_correct += 1
    return float(num_correct) / num_total



def recall_at_k(labels, scores, k=1, doc_num=10):
    scores = scores.reshape(-1, doc_num) # [batch, doc_num]
    labels = labels.reshape(-1, doc_num) # # [batch, doc_num]
    sorted, indices = numpy.sort(scores, 1), numpy.argsort(-scores, 1)
    count_nonzero = 0
    recall = 0
    for i in range(indices.shape[0]):
        num_rel = numpy.sum(labels[i])
        if num_rel==0: continue
        rel = 0
        for j in range(k):
            if labels[i, indices[i, j]] == 1:
                rel += 1
        recall += float(rel) / float(num_rel)
        count_nonzero += 1
    return float(recall) / count_nonzero


def precision_at_k(labels, scores, k=1, doc_num=10):
    
    scores = scores.reshape(-1,doc_num) # [batch, doc_num]
    labels = labels.reshape(-1,doc_num) # [batch, doc_num]

    sorted, indices = numpy.sort(scores, 1), numpy.argsort(-scores, 1)
    count_nonzero = 0
    precision = 0
    for i in range(indices.shape[0]):
        num_rel = numpy.sum(labels[i])
        if num_rel==0: continue
        rel = 0
        for j in range(k):
            if labels[i, indices[i, j]] == 1:
                rel += 1
        precision += float(rel) / float(k)
        count_nonzero += 1
    return precision / count_nonzero


def MAP(target, logits, k=10):
    """
    Compute mean average precision.
    :param target: 2d array [batch_size x num_clicks_per_query] true
    :param logits: 2d array [batch_size x num_clicks_per_query] pred
    :return: mean average precision [a float value]
    """
    assert logits.shape == target.shape

    target = target.reshape(-1,k)
    logits = logits.reshape(-1,k)
    
    sorted, indices = numpy.sort(logits, 1)[::-1], numpy.argsort(-logits, 1)
    count_nonzero = 0
    map_sum = 0
    for i in range(indices.shape[0]):
        average_precision = 0
        num_rel = 0
        for j in range(indices.shape[1]):
            if target[i, indices[i, j]] == 1:
                num_rel += 1
                average_precision += float(num_rel) / (j + 1)
        if num_rel==0: continue
        average_precision = average_precision / num_rel
        # print("average_precision: ", average_precision)
        map_sum += average_precision
        count_nonzero += 1
    #return map_sum / indices.shape[0]
    return float(map_sum) / count_nonzero


def MRR(target, logits, k=10):
    """
    Compute mean reciprocal rank.
    :param target: 2d array [batch_size x rel_docs_per_query]
    :param logits: 2d array [batch_size x rel_docs_per_query]
    :return: mean reciprocal rank [a float value]
    """
    assert logits.shape == target.shape
    target = target.reshape(-1,k)
    logits = logits.reshape(-1,k)

    sorted, indices = numpy.sort(logits, 1)[::-1], numpy.argsort(-logits, 1)
    count_nonzero=0
    reciprocal_rank = 0
    for i in range(indices.shape[0]):
        flag=0
        for j in range(indices.shape[1]):
            if target[i, indices[i, j]] == 1:
                reciprocal_rank += float(1.0) / (j + 1)
                flag=1
                break
        if flag: count_nonzero += 1

    #return reciprocal_rank / indices.shape[0]
    return float(reciprocal_rank) / count_nonzero


def NDCG(target, logits, k):
    """
    Compute normalized discounted cumulative gain.
    :param target: 2d array [batch_size x rel_docs_per_query]
    :param logits: 2d array [batch_size x rel_docs_per_query]
    :return: mean average precision [a float value]
    """
    assert logits.shape == target.shape
    target = target.reshape(-1,k)
    logits = logits.reshape(-1,k)

    assert logits.shape[1] >= k, 'NDCG@K cannot be computed, invalid value of K.'

    sorted, indices = numpy.sort(logits, 1)[::-1], numpy.argsort(-logits, 1)
    NDCG = 0
    for i in range(indices.shape[0]):
        DCG_ref = 0
        num_rel_docs = numpy.count_nonzero(target[i])
        for j in range(indices.shape[1]):
            if j == k:
                break
            if target[i, indices[i, j]] == 1:
                DCG_ref += float(1.0) / numpy.log2(j + 2)
        DCG_gt = 0
        for j in range(num_rel_docs):
            if j == k:
                break
            DCG_gt += float(1.0) / numpy.log2(j + 2)
        NDCG += DCG_ref / DCG_gt

    return float(NDCG) / indices.shape[0]




if __name__=="__main__":
    pass
