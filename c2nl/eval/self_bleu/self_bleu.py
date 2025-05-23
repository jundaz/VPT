from c2nl.eval.bleu import compute_bleu
import numpy as np


def self_bleu_score(sentences):
    '''
    sentences - list of sentences generated by NLG system
    '''
    if len(sentences) == 0:
        return 0
    if len(sentences) == 1:
        return 1
    bleu_scores = []
    sentences = list(sentences)
    for i in range(len(sentences)):
        hyp = sentences[i].split()
        reference = sentences[:i] + sentences[i+1:]
        ref = [i.split() for i in reference]
        bleu_scores.append(compute_bleu([ref], [hyp], smooth=True)[0])

    return np.mean(bleu_scores)