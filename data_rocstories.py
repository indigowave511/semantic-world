"""
ROCStories -> PSF pool, same contract as four_tensor_data_propara.py.

ProPara 학습이 실제로 쓰는 것은 엔티티 그리드가 아니라
문장 시퀀스를 토큰 슬롯 [T+1, N] 으로 만든 것이다. ROCStories 는 스토리마다
문장이 항상 5개(T_max=5)이므로, 같은 encode / rolling window / jsonl 저장을 쓴다.

  unlabeled (spring2016 + winter2017) -> split=train
  cloze val/test (정답 ending 을 sentence5 로 복원) -> split=test
    (four_tensor_train.py 는 train+dev 학습, test 검증)

엔티티 주석이 없으므로 min_entities / entity_states 는 없다.
min_sentences=6 필터도 쓰지 않는다 (적용하면 전량 탈락).

실행:
    python four_tensor_data_rocstories.py
    python four_tensor_data_rocstories.py --T 4 --max_x 16 --out_dir rocstories_T4max_x16
    python four_tensor_data_rocstories.py --no_cloze --unlabeled_test_frac 0.05

산출물:
    psf_pool_T{T}_N{N}.jsonl
    psf_pool_T{T}_N{N}.vocab.json
    psf_pool_T{T}_N{N}.meta.json
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z']+|[.,;]")
T_MAX = 5


def tokenize(sentence: str):
    return TOKEN_RE.findall(sentence.lower())


def build_vocab(stories, min_freq: int = 1):
    counter = Counter()
    for p in stories:
        for s in p["sentences"]:
            counter.update(tokenize(s))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for tok, freq in counter.most_common():
        if freq >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode(sentence: str, vocab: dict, N: int):
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokenize(sentence)]
    ids = ids[:N]
    ids = ids + [vocab["<PAD>"]] * (N - len(ids))
    return ids


def make_rolling_windows(p, vocab, N, T):
    T_max = p["n_sentences"]
    if T >= T_max:
        raise ValueError(f"T={T} >= T_max={T_max}: ROCStories 는 T_max=5, T 는 1..4")
    sents = p["sentences"]
    windows = []
    for s in range(0, T_max - T):
        step_ids_0idx = list(range(s, s + T + 1))
        states = np.stack(
            [encode(sents[i], vocab, N) for i in step_ids_0idx], axis=0
        )
        windows.append({
            "split": p["split"],
            "para_id": p["para_id"],
            "storyid": p["para_id"],
            "source": p["source"],
            "T": T,
            "T_max": T_max,
            "start_0idx": s,
            "raw_sentences": [sents[i] for i in step_ids_0idx],
            "states": states,
        })
    return windows


def read_csv_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_csvs(data_dir: Path):
    unlabeled = sorted(
        p for p in data_dir.glob("*.csv") if p.name.startswith("ROCStories")
    )
    cloze_val = sorted(data_dir.glob("cloze_test_val*.csv"))
    cloze_test = sorted(data_dir.glob("cloze_test_test*.csv"))
    return unlabeled, cloze_val, cloze_test


def load_unlabeled_story(row: dict, source: str):
    sents = [row[f"sentence{i}"] for i in range(1, 6)]
    if any(s is None or str(s).strip() == "" for s in sents):
        return None
    return {
        "split": "train",
        "para_id": row["storyid"],
        "source": source,
        "sentences": sents,
        "n_sentences": T_MAX,
    }


def cloze_gold_sentences(row: dict):
    ctx = [row[f"InputSentence{i}"] for i in range(1, 5)]
    a = row["RandomFifthSentenceQuiz1"]
    b = row["RandomFifthSentenceQuiz2"]
    ans = int(row["AnswerRightEnding"])
    if ans == 1:
        ending = a
    elif ans == 2:
        ending = b
    else:
        raise ValueError(f"AnswerRightEnding={ans!r} (story {row.get('InputStoryid')})")
    return ctx + [ending]


def load_cloze_story(row: dict, source: str, split: str):
    if "AnswerRightEnding" not in row or str(row.get("AnswerRightEnding") or "").strip() == "":
        return None
    sents = cloze_gold_sentences(row)
    if any(s is None or str(s).strip() == "" for s in sents):
        return None
    return {
        "split": split,
        "para_id": row["InputStoryid"],
        "source": source,
        "sentences": sents,
        "n_sentences": T_MAX,
    }


def assign_unlabeled_holdout(stories, frac: float, seed: int):
    """Move a deterministic fraction of unlabeled stories to split=test."""
    if frac <= 0:
        return
    if not 0 < frac < 1:
        raise ValueError(f"unlabeled_test_frac must be in (0, 1), got {frac}")
    unlabeled = [p for p in stories if p["split"] == "train"]
    n_move = int(round(len(unlabeled) * frac))
    ranked = sorted(
        unlabeled,
        key=lambda p: hashlib.md5(
            f"{seed}:{p['para_id']}".encode("utf-8")
        ).hexdigest(),
    )
    moved = {p["para_id"] for p in ranked[:n_move]}
    for p in stories:
        if p["para_id"] in moved:
            p["split"] = "test"
            p["source"] = p["source"] + "_holdout"


def summarize(name, values):
    values = sorted(values)
    n = len(values)
    print(
        f"  [{name}] n={n} min={values[0]} median={values[n // 2]} "
        f"max={values[-1]} mean={sum(values) / n:.2f}"
    )


def build_pool(stories, vocab, N, T):
    pool = []
    for p in stories:
        pool.extend(make_rolling_windows(p, vocab, N, T))
    shapes = set(tuple(w["states"].shape) for w in pool)
    if len(shapes) != 1:
        raise RuntimeError(f"윈도우 shape이 통일되지 않음: {shapes}")
    return pool, shapes.pop()


def main():
    ap = argparse.ArgumentParser(
        description="ROCStories -> PSF pool (ProPara 와 같은 [T+1, N] jsonl)"
    )
    ap.add_argument("--data_dir", type=Path, default=Path("rocstories"))
    ap.add_argument("--T", type=int, default=4, help="rolling window 길이. T_max=5 이므로 1..4. 기본 4=스토리 전체.")
    ap.add_argument(
        "--max_x", type=int, default=16,
        help="문장 내 토큰 슬롯 N. 기본 16 (p99, 절단 <0.5%%, PAD ~39%%). "
             "20 은 unlabeled 최대 길이와 같지만 PAD가 51%%.",
    )
    ap.add_argument("--min_freq", type=int, default=1)
    ap.add_argument("--out_dir", type=Path, default=Path("rocstories_T4max_x16"))
    ap.add_argument(
        "--no_cloze", action="store_true",
        help="cloze CSV 를 넣지 않는다. 이 경우 unlabeled_test_frac>0 이 필요하다.",
    )
    ap.add_argument(
        "--unlabeled_test_frac", type=float, default=0.0,
        help="unlabeled 스토리 중 이 비율을 storyid 해시로 test 로 보낸다.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max_unlabeled", type=int, default=None,
        help="디버그용. unlabeled 스토리 상한 (파일 앞에서 자름).",
    )
    args = ap.parse_args()

    if args.T < 1 or args.T >= T_MAX:
        raise SystemExit(f"T 는 1..{T_MAX - 1} 이어야 한다 (T_max={T_MAX}). got {args.T}")

    data_dir = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(f"data_dir 이 없습니다: {data_dir}")

    unlabeled_paths, cloze_val_paths, cloze_test_paths = find_csvs(data_dir)
    if not unlabeled_paths:
        raise SystemExit(f"ROCStories*.csv 가 없습니다: {data_dir}")

    stories = []
    seen = set()
    skipped_empty = 0
    skipped_dup = 0
    skipped_nolabel = 0

    for path in unlabeled_paths:
        rows = read_csv_rows(path)
        if args.max_unlabeled is not None:
            rows = rows[: max(0, args.max_unlabeled - sum(
                1 for p in stories if p["split"] == "train"
            ))]
        for row in rows:
            p = load_unlabeled_story(row, source=path.name)
            if p is None:
                skipped_empty += 1
                continue
            if p["para_id"] in seen:
                skipped_dup += 1
                continue
            seen.add(p["para_id"])
            stories.append(p)

    n_unlabeled = len(stories)

    if not args.no_cloze:
        if not cloze_val_paths and not cloze_test_paths:
            raise SystemExit(f"cloze_test_*.csv 가 없습니다: {data_dir}")
        for path in cloze_val_paths + cloze_test_paths:
            rows = read_csv_rows(path)
            if rows and "AnswerRightEnding" not in rows[0]:
                print(f"[skip] {path.name}: AnswerRightEnding 컬럼 없음 (정답 ending 복원 불가)")
                skipped_nolabel += len(rows)
                continue
            for row in rows:
                p = load_cloze_story(row, source=path.name, split="test")
                if p is None:
                    if "AnswerRightEnding" not in row or str(row.get("AnswerRightEnding") or "").strip() == "":
                        skipped_nolabel += 1
                    else:
                        skipped_empty += 1
                    continue
                if p["para_id"] in seen:
                    skipped_dup += 1
                    continue
                seen.add(p["para_id"])
                stories.append(p)

    assign_unlabeled_holdout(stories, args.unlabeled_test_frac, args.seed)

    n_train = sum(1 for p in stories if p["split"] == "train")
    n_test = sum(1 for p in stories if p["split"] == "test")
    if n_train == 0:
        raise SystemExit("train 스토리가 0개입니다.")
    if n_test == 0:
        raise SystemExit(
            "test 스토리가 0개입니다. cloze 를 쓰거나 --unlabeled_test_frac 를 지정하세요."
        )

    print(f"[load] unlabeled csv={len(unlabeled_paths)}  cloze_val={len(cloze_val_paths)}  "
          f"cloze_test={len(cloze_test_paths)}")
    print(f"[load] unlabeled stories={n_unlabeled}  total stories={len(stories)}  "
          f"empty_skip={skipped_empty} dup_skip={skipped_dup} nolabel_skip={skipped_nolabel}")
    print(f"[split] train={n_train}  test={n_test}  "
          f"(train.py: train 학습, test 검증)")
    summarize("문장수/스토리", [p["n_sentences"] for p in stories])

    vocab = build_vocab(stories, min_freq=args.min_freq)
    N_auto = max(len(tokenize(s)) for p in stories for s in p["sentences"])
    N = args.max_x if args.max_x is not None else N_auto

    sent_lens = [len(tokenize(s)) for p in stories for s in p["sentences"]]
    n_truncated = sum(1 for L in sent_lens if L > N)
    pad_ratio = 1 - sum(min(L, N) for L in sent_lens) / (len(sent_lens) * N)
    print(f"\n[vocab] size={len(vocab)}  min_freq={args.min_freq}")
    print(f"[vocab] N(=max_x) = {N}"
          + (f"  (자동계산값 {N_auto} 대신 명시적 지정)" if args.max_x is not None
             else "  (서브셋 기준 자동계산)"))
    print(f"[vocab] 잘리는 문장: {n_truncated}/{len(sent_lens)}개 "
          f"({n_truncated / len(sent_lens):.1%}),  예상 PAD 비율: {pad_ratio:.1%}")

    pool, window_shape = build_pool(stories, vocab, N, args.T)
    n_train_w = sum(1 for w in pool if w["split"] == "train")
    n_test_w = sum(1 for w in pool if w["split"] == "test")
    print(f"\n[pool] T={args.T} 고정, T_max={T_MAX}, 스토리당 윈도우 {T_MAX - args.T}개")
    print(f"[pool] 윈도우 수: {len(pool)}  (train={n_train_w}, test={n_test_w})")
    print(f"[pool] 모든 윈도우 shape 동일: {window_shape}  == [T+1, N] == "
          f"[{args.T + 1}, {N}]")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"T{args.T}_N{N}"
    out_path = args.out_dir / f"psf_pool_{tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for w in pool:
            rec = dict(w)
            rec["states"] = rec["states"].tolist()
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    vocab_path = args.out_dir / f"psf_pool_{tag}.vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=1)

    by_source = Counter(p["source"] for p in stories)
    meta = {
        "T": args.T,
        "N": N,
        "N_auto": N_auto,
        "T_max": T_MAX,
        "n_sentences_truncated": n_truncated,
        "pad_ratio": round(pad_ratio, 4),
        "vocab_size": len(vocab),
        "min_freq": args.min_freq,
        "window_shape": list(window_shape),
        "n_stories_total": len(stories),
        "n_stories_train": n_train,
        "n_stories_test": n_test,
        "n_windows": len(pool),
        "n_windows_train": n_train_w,
        "n_windows_test": n_test_w,
        "unlabeled_test_frac": args.unlabeled_test_frac,
        "include_cloze": not args.no_cloze,
        "max_unlabeled": args.max_unlabeled,
        "stories_by_source": dict(by_source),
        "data_dir": str(data_dir),
    }
    meta_path = args.out_dir / f"psf_pool_{tag}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print(f"\n[out] {out_path}")
    print(f"[out] {vocab_path}")
    print(f"[out] {meta_path}")


if __name__ == "__main__":
    main()
