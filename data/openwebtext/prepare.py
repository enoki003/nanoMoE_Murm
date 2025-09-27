"""Download and tokenize OpenWebText-style corpora for nanoGPT.

This script now supports multiple dataset backends because the original
`openwebtext` mirror occasionally disappears from Hugging Face. By default the
script tries the classic raw-text mirror (`Skylion007/openwebtext`) if an
authentication token is available, and otherwise falls back to the public,
pre-tokenised dump (`NeelNanda/openwebtext-tokenized-9b`).

Environment variables:

* ``OPENWEBTEXT_DATASET`` – override the dataset repository id to use.
* ``OPENWEBTEXT_FALLBACK`` – override the fallback dataset repository id.
* ``HF_TOKEN`` – Hugging Face token to access gated datasets (optional).

You can also pass these values via the corresponding CLI flags. The output
format (``train.bin``/``val.bin`` of ``uint16`` tokens) remains identical to
the historical prepare step shipped with nanoGPT.
"""

import argparse
import logging
import os
from typing import Iterable, Optional, Sequence

import numpy as np
import tiktoken
from datasets import load_dataset  # huggingface datasets
from datasets.builder import DatasetGenerationError
from huggingface_hub.errors import RepositoryNotFoundError
from tqdm import tqdm

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = int(os.environ.get("OPENWEBTEXT_NUM_PROC", "8"))

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = int(os.environ.get("OPENWEBTEXT_NUM_PROC_LOAD", str(num_proc)))

enc = tiktoken.get_encoding("gpt2")

DEFAULT_DATASET_ID = "Skylion007/openwebtext"
FALLBACK_DATASET_ID = "NeelNanda/openwebtext-tokenized-9b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OpenWebText-style dataset for nanoGPT")
    parser.add_argument(
        "--dataset-id",
        default=os.environ.get("OPENWEBTEXT_DATASET", DEFAULT_DATASET_ID),
        help="Primary Hugging Face dataset repo id to download (raw text expected).",
    )
    parser.add_argument(
        "--fallback-id",
        default=os.environ.get("OPENWEBTEXT_FALLBACK", FALLBACK_DATASET_ID),
        help="Fallback dataset repo id to try when the primary one is unavailable.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face access token for gated datasets.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=float(os.environ.get("OPENWEBTEXT_VAL_RATIO", "0.0005")),
        help="Fraction of examples reserved for validation.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=num_proc,
        help="Multiprocessing workers for tokenisation steps.",
    )
    parser.add_argument(
        "--num-proc-load",
        type=int,
        default=num_proc_load_dataset,
        help="Multiprocessing workers for dataset download/prepare stage.",
    )
    parser.add_argument(
        "--force-fallback",
        action="store_true",
        help="Skip the primary dataset and immediately use the fallback.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Limit the number of documents processed (useful for quick smoke tests).",
    )
    return parser.parse_args()


def _load_dataset(repo_id: str, token: Optional[str], num_workers: int):
    logging.info("Loading dataset: %s", repo_id)
    load_kwargs = {"num_proc": num_workers}
    if token:
        load_kwargs["token"] = token
    return load_dataset(repo_id, **load_kwargs)


def _tokenise_text_column(dataset_dict, num_workers: int):
    def process(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(enc.eot_token)
        return {"ids": ids, "len": len(ids)}

    return dataset_dict.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_workers,
    )


def _prepare_tokenised_column(dataset_dict):
    def process(example):
        ids = example["tokens"]
        if isinstance(ids, np.ndarray):
            ids = ids.astype(np.uint16).tolist()
        return {"ids": ids, "len": len(ids)}

    return dataset_dict.map(
        process,
        remove_columns=["tokens"],
        desc="formatting pre-tokenized splits",
    )


def _ensure_uint16(values: Sequence[int]) -> Iterable[np.uint16]:
    arr = np.asarray(values, dtype=np.uint32)
    if arr.size and arr.max() >= 2**16:
        raise ValueError("Encountered token id >= 65536; cannot store in uint16.")
    return arr.astype(np.uint16)


def _materialise_bin_files(tokenised):
    for split, dset in tokenised.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
        dtype = np.uint16  # GPT-2 vocab fits into uint16
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = min(1024, max(1, len(dset)))

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = (
                dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True)
                .with_format("numpy")
            )
            ids_list = batch["ids"]
            if len(ids_list) == 0:
                continue
            arr_batch = np.concatenate([_ensure_uint16(ids) for ids in ids_list])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    attempted_sources = []
    dataset = None
    used_repo_id: Optional[str] = None

    candidate_ids = []
    if args.force_fallback:
        candidate_ids.append(args.fallback_id)
    else:
        candidate_ids.extend([args.dataset_id, args.fallback_id])

    candidate_ids = [cid for cid in candidate_ids if cid]
    candidate_ids = list(dict.fromkeys(candidate_ids))

    for repo_id in candidate_ids:
        if repo_id in attempted_sources:
            continue
        attempted_sources.append(repo_id)
        try:
            dataset = _load_dataset(repo_id, args.hf_token, args.num_proc_load)
            used_repo_id = repo_id
            break
        except (FileNotFoundError, RepositoryNotFoundError, DatasetGenerationError, ValueError) as exc:
            logging.warning("Failed to load %s: %s", repo_id, exc)
    if dataset is None:
        raise RuntimeError(
            "Unable to download any OpenWebText dataset. "
            "Please provide a valid dataset id via --dataset-id or OPENWEBTEXT_DATASET."
        )

    logging.info("Loaded dataset splits: %s", list(dataset.keys()))

    if args.max_docs is not None:
        max_docs = max(1, args.max_docs)
        available = len(dataset["train"])
        if max_docs < available:
            logging.info("Selecting first %d documents out of %d for debugging", max_docs, available)
            dataset["train"] = dataset["train"].select(range(max_docs))

    if "train" not in dataset:
        raise ValueError("Expected a 'train' split in the dataset.")

    split_dataset = dataset["train"].train_test_split(
        test_size=args.val_ratio, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")

    if "text" in split_dataset["train"].column_names:
        tokenized = _tokenise_text_column(split_dataset, args.num_proc)
    elif "tokens" in split_dataset["train"].column_names:
        tokenized = _prepare_tokenised_column(split_dataset)
    else:
        raise ValueError(
            "Unsupported dataset schema. Expected a 'text' or 'tokens' column."
        )

    _materialise_bin_files(tokenized)

    logging.info(
        "Finished writing train.bin and val.bin from dataset %s (fallback=%s)",
        used_repo_id,
        attempted_sources[-1] if len(attempted_sources) > 1 else "n/a",
    )
