"""
Text similarity metrics for qualitative evaluation.

Vendored from Fox benchmark (https://github.com/ucaslcl/Fox)
Original file: eval_tools/eval_ocr_test.py
"""

import re

import jieba
import nltk
from nltk.metrics import f_measure, precision, recall
from nltk.translate import meteor_score


def contain_chinese_string(text):
    """Check if text contains Chinese characters."""
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(chinese_pattern.search(text))


def cal_per_metrics(predict_root_, pred, gt):
    """
    Calculate text similarity metrics between prediction and ground truth.

    Args:
        predict_root_: Unused (kept for API compatibility)
        pred: Predicted/generated text
        gt: Ground truth text

    Returns:
        Dict with keys: bleu, meteor, f_measure, precision, recall, edit_dist
    """
    metrics = {}

    if contain_chinese_string(gt) or contain_chinese_string(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)

    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["f_measure"] = f_measure(reference, hypothesis)

    metrics["precision"] = precision(reference, hypothesis)
    metrics["recall"] = recall(reference, hypothesis)
    metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))

    return metrics
