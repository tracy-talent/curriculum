from typing import Dict, List
from sklearn.metrics import f1_score

import argparse


def str_to_dict(prediction: str) -> Dict[int, int]:
    result: Dict[int, int] = {}
    with open(prediction) as fh:
        for line in fh:
            id_, label = line.split('\t')
            assert int(id_) not in result, "duplicated prediction"
            assert int(label) in (0, 1), "invalid label"
            result[int(id_)] = int(label)
    return result


def metrics(prediction: Dict[int, int], ground_truth: Dict[int, int]):
    assert len(set(prediction.keys()) ^ set(ground_truth.keys())) == 0
    y_true, y_pred = [], []
    for id_ in ground_truth.keys():
        y_true.append(ground_truth[id_])
        y_pred.append(prediction[id_])
    return f1_score(y_true, y_pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Compute Metrics",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ground_truth", required=True,
                        help="data to ground truth")
    parser.add_argument("--prediction", required=True,
                        help="data to prediction")
    args = parser.parse_args()
    ground_truth = str_to_dict(args.ground_truth)
    prediction = str_to_dict(args.prediction)
    print(metrics(prediction, ground_truth))
