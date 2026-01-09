from typing import List

from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider


def compute_rouge_l(references: List[str], hypotheses: List[str]) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for ref, hyp in zip(references, hypotheses):
        score = scorer.score(ref, hyp)["rougeL"].fmeasure
        scores.append(score)
    return float(sum(scores) / max(len(scores), 1))


def compute_meteor(references: List[str], hypotheses: List[str]) -> float:
    scores = []
    for ref, hyp in zip(references, hypotheses):
        scores.append(meteor_score([ref.split()], hyp.split()))
    return float(sum(scores) / max(len(scores), 1))


def compute_cider(references: List[str], hypotheses: List[str]) -> float:
    """
    使用 pycocoevalcap 计算 CIDEr-D。
    输入：等长的 ref/hyp 列表。
    """
    assert len(references) == len(hypotheses), "refs 与 hyps 数量需一致"
    gts = {i: [ref] for i, ref in enumerate(references)}
    res = {i: [hypotheses[i]] for i in range(len(hypotheses))}
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts, res)
    return float(score)


