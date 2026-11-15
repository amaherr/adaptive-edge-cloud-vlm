"""Run the pinned upstream VQA evaluator in a Python 2 environment.

The upstream GT-Vision-Lab/VQA master sources use Python 2 syntax. Keep this
small adapter compatible with both Python 2.7 and Python 3; it invokes the
upstream evaluator without reimplementing its answer normalization.
"""

import argparse
import hashlib
import json
import os
import sys


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqa-root", required=True)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if os.path.exists(args.output):
        parser.error("refusing to overwrite an official evaluation")
    with open(args.annotations) as handle:
        annotations = json.load(handle)["annotations"]
    with open(args.predictions) as handle:
        predictions = json.load(handle)
    expected = [entry["question_id"] for entry in annotations]
    actual = [entry["question_id"] for entry in predictions]
    if len(expected) != len(set(expected)) or sorted(expected) != sorted(actual):
        parser.error("predictions must cover every subset annotation exactly once")

    sys.path.insert(0, os.path.join(args.vqa_root, "PythonHelperTools", "vqaTools"))
    sys.path.insert(0, os.path.join(args.vqa_root, "PythonEvaluationTools"))
    from vqa import VQA
    from vqaEvaluation.vqaEval import VQAEval

    vqa = VQA(args.annotations, args.questions)
    results = vqa.loadRes(args.predictions, args.questions)
    evaluator = VQAEval(vqa, results)
    evaluator.evaluate()
    payload = {
        "schema_version": 1,
        "evaluator": "GT-Vision-Lab/VQA/PythonEvaluationTools",
        "question_count": len(expected),
        "accuracy_percent": evaluator.accuracy,
        "per_question_accuracy_percent": evaluator.evalQA,
        "input_sha256": {
            "questions": _sha256(args.questions),
            "annotations": _sha256(args.annotations),
            "predictions": _sha256(args.predictions),
        },
        "evaluator_source_sha256": {
            "vqa.py": _sha256(
                os.path.join(args.vqa_root, "PythonHelperTools", "vqaTools", "vqa.py")
            ),
            "vqaEval.py": _sha256(
                os.path.join(
                    args.vqa_root,
                    "PythonEvaluationTools",
                    "vqaEvaluation",
                    "vqaEval.py",
                )
            ),
        },
    }
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("Official VQA accuracy: {}%".format(evaluator.accuracy["overall"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
