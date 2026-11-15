"""Inspect answer normalization using the pinned upstream routines (Python 2/3).

This is diagnostic output; official primary scores come from VQAEval.evaluate.
"""

import argparse
import hashlib
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqa-root", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, os.path.join(args.vqa_root, "PythonEvaluationTools"))
    from vqaEvaluation.vqaEval import VQAEval

    # Construct with a minimal interface, so the unmodified initializer owns all
    # normalization tables. No alternate contractions or article rules are used.
    class EmptyVQA:
        def getQuesIds(self):
            return []

    normalizer = VQAEval(EmptyVQA(), EmptyVQA())
    with open(args.annotations) as stream:
        annotations = json.load(stream)["annotations"]
    with open(args.predictions) as stream:
        predictions = {r["question_id"]: r["answer"] for r in json.load(stream)}

    def cleanup(text):
        return text.replace("\n", " ").replace("\t", " ").strip()

    def normalized(text):
        return normalizer.processDigitArticle(normalizer.processPunctuation(text))

    result = {}
    for annotation in annotations:
        qid = annotation["question_id"]
        refs = [cleanup(a["answer"]) for a in annotation["answers"]]
        answer = cleanup(predictions[qid])
        branch = len(set(refs)) > 1
        result[qid] = {
            "official_normalization_branch_active": branch,
            "prediction_after_actual_official_branch": normalized(answer)
            if branch
            else answer,
            "references_after_actual_official_branch": [normalized(r) for r in refs]
            if branch
            else refs,
            "prediction_unconditionally_normalized_for_diagnosis": normalized(answer),
            "references_unconditionally_normalized_for_diagnosis": [
                normalized(r) for r in refs
            ],
        }
    with open(__file__, "rb") as source:
        inspector_sha = hashlib.sha256(source.read()).hexdigest()
    with open(args.output, "w") as stream:
        json.dump(
            {
                "questions": result,
                "inspector_sha256": inspector_sha,
            },
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
