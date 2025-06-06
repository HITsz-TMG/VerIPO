import re
import argparse
import json
import numpy as np
from tqdm import tqdm
from math_verify import ExprExtractionConfig, LatexExtractionConfig, StringExtractionConfig, parse, verify
from transformers import AutoTokenizer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pred_path", type=str, default="")
args = parser.parse_args()

data = json.load(open(args.pred_path, "r"))
acc = 0

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


choices = ['a','b','c','d', 'e', 'f', "g", 'h', 'i', 'j']
for line in tqdm(data):
    ground_truth = str(line["ground_truth"])
    output = line["model_output"]
    
    model_output = line["model_output"]
    pattern = (
        r"^(?=(?:.*<think>){1})(?=(?:.*<\/think>){1})"
        r"(?=(?:.*<answer>){1})(?=(?:.*<\/answer>){1})"
        r"(?!.*<think>.*<think>)"
        r"(?!.*<\/think>.*<\/think>)"
        r"(?!.*<answer>.*<answer>)"
        r"(?!.*<\/answer>.*<\/answer>)"
        r".*<think>(.+?)</think>\s*<answer>.+?</answer>.*$"
    )
    matches = re.search(pattern, model_output, re.DOTALL)
    
    # For Multi-Choices
    if ground_truth.isalpha() and ground_truth.isupper():
        content_match = re.search(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
        student_answer = content_match.group(1).strip() if content_match else ""
        student_answer = student_answer.replace('</answer>','').replace('<answer>','').strip()
        line["extracted_answer"] = student_answer
        if str(ground_truth).lower() in student_answer.lower():
            choices_other = [choice for choice in choices if choice != str(ground_truth).lower()]
            if all(choice not in student_answer.lower() for choice in choices_other):
                acc += 1
                line["correct"] = True
    else: # For Numerical
        if "vsi_bench" in args.pred_path: # Mean Relative Accuracy
            content_match = re.search(r'<answer>(.*?)</answer>', model_output, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else ""
            line["extracted_answer"] = student_answer
            if student_answer == "":
                continue
            student_answer = student_answer.replace('</answer>','').replace('<answer>','').strip().strip("$")
            try:
                acc += mean_relative_accuracy(to_float(fuzzy_matching(student_answer)), to_float(ground_truth.strip("$").strip()), start=.5, end=.95, interval=.05)
            except:
                continue
        else: # Math Verify
            if ground_truth == "['24/7', '3.429', '3.43]" or ground_truth == "['24/7', '3.429', '3.43']":
                ground_truth = "24/7"
            sol = f"${str(ground_truth)}$"
            gold_parsed = parse(sol)
            answer_parsed = parse(
                model_output,
                extraction_config=[StringExtractionConfig(), LatexExtractionConfig(), ExprExtractionConfig()],
            )
            acc += float(verify(answer_parsed, gold_parsed))
    
    if "image" in line["question"]: 
        del line["question"]["image"]

print(f"Accuracy: {acc / len(data) * 100:.2f}%")
# json.dump(data, open(pred_path, "w"), indent=2)
