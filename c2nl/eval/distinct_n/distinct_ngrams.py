from c2nl.eval.distinct_n.utils import ngrams


def distinct_n_corpus_level(sentences, n=1):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float: total distinct ngrams / total ngrams.
    """
    all_ngrams = []
    distinct_ngrams = set()
    for sentence in sentences:
        total_ngrams = list(ngrams(sentence, n))
        all_ngrams.extend(total_ngrams)
        distinct_ngrams.update(total_ngrams)
    if len(all_ngrams) == 0:
        print("Warning: The corpus is empty, score is zero.")
        return 0
    return len(distinct_ngrams) / len(all_ngrams)
