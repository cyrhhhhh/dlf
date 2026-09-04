"""Convert the extracted MOSI utterance features to the DLF dataset format."""

import argparse
import pickle
from pathlib import Path

import numpy as np


FEATURES = {
    "text": ("deberta-large-4-UTT", 1024),
    "audio": ("wav2vec-large-c-UTT", 512),
    "vision": ("manet_UTT", 1024),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).parent / "dataset" / "MOSI",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent
        / "dataset"
        / "MOSI"
        / "Processed"
        / "utterance_2way.pkl",
    )
    return parser.parse_args()


def load_feature(path, expected_dim):
    feature = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    if feature.shape != (expected_dim,):
        raise ValueError(
            f"Unexpected feature shape for {path}: {feature.shape}, "
            f"expected ({expected_dim},)"
        )
    if not np.isfinite(feature).all():
        raise ValueError(f"Non-finite values found in {path}")
    return feature


def build_split(video_ids, utterance_ids, labels, raw_text, features_root):
    split = {name: [] for name in FEATURES}
    split.update({"raw_text": [], "id": [], "regression_labels": []})

    for video_id in sorted(video_ids):
        ids = utterance_ids[video_id]
        video_labels = labels[video_id]
        video_text = raw_text[video_id]
        if not (len(ids) == len(video_labels) == len(video_text)):
            raise ValueError(f"Mismatched utterance metadata for video {video_id}")

        for index, utterance_id in enumerate(ids):
            for modality, (directory, expected_dim) in FEATURES.items():
                feature_path = features_root / directory / f"{utterance_id}.npy"
                split[modality].append(load_feature(feature_path, expected_dim))
            split["raw_text"].append(str(video_text[index]))
            split["id"].append(utterance_id)
            split["regression_labels"].append(float(video_labels[index]))

    for modality in FEATURES:
        split[modality] = np.stack(split[modality])[:, np.newaxis, :]
    split["raw_text"] = np.asarray(split["raw_text"])
    split["id"] = np.asarray(split["id"])
    split["regression_labels"] = np.asarray(
        split["regression_labels"], dtype=np.float32
    )
    return split


def main():
    args = parse_args()
    source_path = args.dataset_dir / "CMUMOSI_features_raw_2way.pkl"
    with source_path.open("rb") as handle:
        payload = pickle.load(handle)

    if not isinstance(payload, list) or len(payload) != 7:
        raise ValueError(
            f"Unexpected source format in {source_path}: expected a 7-item list"
        )

    utterance_ids, labels, _, raw_text, train_ids, valid_ids, test_ids = payload
    split_ids = {
        "train": train_ids,
        "valid": valid_ids,
        "test": test_ids,
    }
    assigned_ids = set().union(*split_ids.values())
    if assigned_ids != set(utterance_ids):
        raise ValueError("Train/valid/test video IDs do not cover the dataset exactly")

    features_root = args.dataset_dir / "features"
    dataset = {
        name: build_split(
            video_ids, utterance_ids, labels, raw_text, features_root
        )
        for name, video_ids in split_ids.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {args.output}")
    for name, split in dataset.items():
        print(
            f"{name}: {len(split['id'])} samples, "
            f"text={split['text'].shape}, audio={split['audio'].shape}, "
            f"vision={split['vision'].shape}"
        )


if __name__ == "__main__":
    main()