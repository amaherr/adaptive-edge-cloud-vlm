"""Run synthetic checks against the unmodified official evaluator (Python 2).

Usage: PYTHON2 check_upstream_cases.py --vqa-root PINNED_CHECKOUT
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqa-root", required=True)
    args = parser.parse_args()
    sys.path.insert(0, args.vqa_root + "/PythonEvaluationTools")
    from vqaEvaluation.vqaEval import VQAEval

    class Dataset:
        def __init__(self, qa):
            self.qa = qa

        def getQuesIds(self):
            return sorted(self.qa)

    cases = [
        ("Hotel", ["hotel"] * 10, 0.0),
        ("hotel", ["hotel"] * 10, 100.0),
        ("Hotel", ["hotel"] * 9 + ["motel"], 100.0),
        ("two", ["2"] * 9 + ["3"], 100.0),
        ("There are two people.", ["2"] * 9 + ["3"], 0.0),
        ("yes", ["yes"] * 1 + ["no"] * 9, 30.0),
        ("yes", ["yes"] * 2 + ["no"] * 8, 60.0),
        ("yes", ["yes"] * 3 + ["no"] * 7, 90.0),
        ("yes", ["yes"] * 4 + ["no"] * 6, 100.0),
    ]
    for prediction, references, expected in cases:
        question = {
            "question_type": "test",
            "answer_type": "other",
            "answers": [
                {"answer": answer, "answer_id": i + 1}
                for i, answer in enumerate(references)
            ],
        }
        evaluator = VQAEval(
            Dataset({1: question}), Dataset({1: {"answer": prediction}})
        )
        evaluator.evaluate()
        actual = evaluator.evalQA[1]
        assert actual == expected, (prediction, references, actual, expected)
    print("9 official evaluator cases passed")


if __name__ == "__main__":
    main()
