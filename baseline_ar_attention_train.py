import argparse
import json
import logging
import os
import socket
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader

import four_tensor_train as ftt
from four_tensor_train import (
    ProParaDataset,
    _bool_arg,
    _ce_ppl,
    _git_note,
    _loader_kwargs,
    _nvidia_driver_version,
    _nvidia_proc_version,
    _nvidia_smi_gpus,
    _repro_env,
    _software_versions,
    _source_file_note,
    fill_runtime_reproducibility,
    find_resume_checkpoint,
    load_and_split_data,
    load_meta,
    load_pool_files,
    optimize_for_rtx3080,
    set_seed_for_reproducibility,
    setup_dual_gpu_model,
    unwrap,
    write_metrics_json,
)
from baseline_ar_attention_model import BaselineARAttention, BaselineARUpdater, compute_d

logger = logging.getLogger("baseline_ar_attention_train")

warnings.filterwarnings(
    "ignore",
    message=r"Deterministic behavior was enabled.*uses CuBLAS and you have CUDA",
    category=UserWarning,
)


def setup_logging(*log_paths: Path):
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handlers = []
    seen = set()
    for log_path in log_paths:
        log_path = Path(log_path)
        key = str(log_path.resolve())
        if key in seen:
            continue
        seen.add(key)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        handlers.append(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    handlers.append(sh)
    for log in (logger, ftt.logger):
        log.handlers.clear()
        log.setLevel(logging.INFO)
        log.propagate = False
        for h in handlers:
            log.addHandler(h)


def collect_provenance() -> dict:
    import getpass
    import platform

    here = Path(__file__).resolve()
    model_src = here.with_name("baseline_ar_attention_model.py")
    smi_gpus = _nvidia_smi_gpus()
    return {
        "machine": {
            "hostname": socket.gethostname(),
            "username": getpass.getuser(),
            "platform": platform.platform(),
            "processor": platform.processor() or None,
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "torch_cuda_built": torch.version.cuda,
        },
        "source_files": {
            "baseline_ar_attention_model.py": _source_file_note(model_src),
            "baseline_ar_attention_train.py": _source_file_note(here),
        },
        "git": _git_note(here.parent),
        "argv": sys.argv,
        "reproducibility": {
            "software": _software_versions(),
            "env": _repro_env(),
            "gpu": {
                "driver_version": _nvidia_driver_version(smi_gpus),
                "nvidia_proc_version": _nvidia_proc_version(),
                "nvidia_smi": smi_gpus,
            },
        },
    }


@torch.no_grad()
def evaluate(model, loader, device):
    """fair CE (incl. EOS) is the train val_loss. S(T) CE/acc drop EOS and PAD."""
    model.eval()
    raw = unwrap(model)
    pin = device.type == "cuda"
    total_fair = 0.0
    n_batches = 0
    n_correct = 0
    n_tok = 0
    st_ce = 0.0
    st_n = 0
    for token_ids_full in loader:
        token_ids_full = token_ids_full.to(device, non_blocking=pin)
        loss_sum, n_all, content_sum, n_content = raw._fair_unroll(token_ids_full)
        hat = raw.generate_S3(token_ids_full)
        gold = raw.gold_S_last(token_ids_full)
        mask = gold != raw.padding_idx
        n_correct += int(((hat == gold) & mask).sum().item())
        n_tok += int(mask.sum().item())
        if n_all > 0:
            total_fair += float(loss_sum.item()) / n_all
        n_batches += 1
        st_ce += float(content_sum.item())
        st_n += int(n_content)
    n_batches = max(n_batches, 1)
    gold_n = max(n_tok, 1)
    content_n = max(st_n, 1)
    return {
        "fair": total_fair / n_batches,
        "s_t_ce": st_ce / content_n,
        "token_acc": n_correct / gold_n,
        "s_t_acc": n_correct / gold_n,
        "s_t_n_tok": n_tok,
        "s_t_n_matched": n_correct,
        "s_t_ce_sum": st_ce,
    }


def save_checkpoint(path: Path, model, updater, epoch, avg_train, avg_val,
                    train_losses, val_losses, best_val_loss, best_val_epoch, config,
                    val_token_accs=None, val_s_t_ces=None):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": unwrap(model).state_dict(),
            "optimizer_state_dict": updater.optimizer.state_dict(),
            "train_loss": avg_train,
            "val_loss": avg_val,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_token_accs": [] if val_token_accs is None else val_token_accs,
            "val_s_t_ces": [] if val_s_t_ces is None else val_s_t_ces,
            "best_val_loss": best_val_loss,
            "best_val_epoch": best_val_epoch,
            "config": config,
        },
        path,
    )


def write_losses_jsonl(path: Path, rows: list, header: dict):
    running_best = float("inf")
    running_best_epoch = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"record": "run_header", **header}) + "\n")
        for rec in rows:
            va = rec["val_loss"]
            if va < running_best:
                running_best = va
                running_best_epoch = rec["epoch"]
            out = {
                "record": "epoch",
                **rec,
                "train_ppl": _ce_ppl(rec["train_loss"]),
                "val_ppl": _ce_ppl(va),
                "gap": va - rec["train_loss"],
                "best_val_loss": running_best,
                "best_val_epoch": running_best_epoch,
            }
            if "val_s_t_ce" in rec:
                out["val_s_t_ppl"] = _ce_ppl(rec["val_s_t_ce"])
            f.write(json.dumps(out) + "\n")


def train_routine(args):
    output_dir = Path(args.output_dir)
    tag = args.run_tag
    cwd_log = Path.cwd() / "baseline_ar_attention_train.log"
    run_log = output_dir / f"{tag}.log"
    if os.environ.get("BASELINE_AR_TERMINAL_LOG") or os.environ.get("FOUR_TENSOR_TERMINAL_LOG"):
        setup_logging(run_log)
    else:
        setup_logging(run_log, cwd_log)
    started = datetime.now(timezone.utc)
    t0 = time.time()
    provenance = collect_provenance()

    set_seed_for_reproducibility(args.seed, deterministic=args.deterministic)
    if torch.cuda.is_available() and not args.deterministic:
        optimize_for_rtx3080()

    data_dir = Path(args.data_dir)
    jsonl_path, meta_path, vocab_path = load_pool_files(data_dir)
    meta = load_meta(meta_path)

    T = int(meta["T"])
    N = int(meta["N"])
    vocab_size = int(meta["vocab_size"])
    batch_size = args.batch_size
    d_x = args.d_x
    d_t = args.d_t
    d = compute_d(d_x, d_t)
    n_layers = args.n_layers
    n_heads = args.n_heads
    epochs = args.epochs
    lr = args.lr

    train_list, val_list = load_and_split_data(jsonl_path)
    logger.info(f"[Run] output_dir : {output_dir}")
    logger.info(f"[Run] file tag   : {tag}")
    logger.info(f"[Run] logs       : {run_log.name}  and  {cwd_log}")
    logger.info(f"[Data] dir        : {data_dir}")
    logger.info(f"[Data] jsonl      : {jsonl_path.name}")
    logger.info(f"[Data] vocab      : {vocab_path.name}")
    logger.info(
        f"[Data] T={T}, N={N}, vocab_size={vocab_size}, windows={meta['n_windows']}"
    )
    logger.info(f"[Data Load] Train 샘플 수 (train+dev): {len(train_list)}")
    logger.info(f"[Data Load] Validation 샘플 수 (test): {len(val_list)}")

    model = BaselineARAttention(
        vocab_size=vocab_size, N=N, T=T, d_x=d_x, d_t=d_t,
        n_layers=n_layers, n_heads=n_heads, dropout=args.dropout,
        padding_idx=args.padding_idx,
    )
    model, device = setup_dual_gpu_model(model)
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 1
    n_gpus = max(n_gpus, 1)
    gpu_names = (
        [torch.cuda.get_device_name(i) for i in range(n_gpus)]
        if device.type == "cuda" else []
    )

    loader_batch = batch_size * n_gpus if isinstance(model, DataParallel) else batch_size
    pin = device.type == "cuda"
    drop_last = isinstance(model, DataParallel)

    train_loader = DataLoader(
        ProParaDataset(train_list, T=T, N=N),
        batch_size=loader_batch,
        shuffle=True,
        drop_last=drop_last,
        **_loader_kwargs(args.num_workers, pin, args.seed, shuffle=True),
    )
    val_loader = DataLoader(
        ProParaDataset(val_list, T=T, N=N),
        batch_size=loader_batch,
        shuffle=False,
        drop_last=False,
        **_loader_kwargs(args.num_workers, pin, args.seed, shuffle=False),
    )

    updater = BaselineARUpdater(
        model, lr=lr, optimizer=args.optimizer,
    )
    raw = unwrap(model)
    n_params = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    logger.info(
        f"[Model] BaselineARAttention on {device}  gpus={n_gpus}  params={n_params:,}"
    )
    logger.info(
        f"[Model] d={d} (d_x={d_x} d_t={d_t}) n_layers={n_layers} n_heads={n_heads} "
        f"dropout={args.dropout} padding_idx={args.padding_idx} "
        f"bos={raw.bos_id} eos={raw.eos_id} vocab_ext={raw.vocab_size} "
        f"train=AR-fair "
        f"batch/gpu={batch_size} loader_batch={loader_batch} "
        f"effective_batch={loader_batch}"
    )
    logger.info(
        "[Train] AR free-running only: own tokens on the left of S(T). "
        "Val loss is the same AR CE. Gen token-acc is the decoded comparison."
    )
    logger.info(f"Random seed: {args.seed} (deterministic mode: {args.deterministic})")
    provenance["machine"]["cuda_available"] = torch.cuda.is_available()
    provenance["machine"]["gpu_names"] = gpu_names
    repro = fill_runtime_reproducibility(provenance, args)
    src = provenance["source_files"]["baseline_ar_attention_model.py"]
    logger.info(
        f"[Provenance] baseline_ar_attention_model.py mtime={src['mtime_utc']} "
        f"sha256={src['sha256'][:12]}..."
    )
    logger.info(
        f"[Provenance] machine={provenance['machine']['hostname']} "
        f"user={provenance['machine']['username']} "
        f"torch={provenance['machine']['torch']}"
    )
    gpu_repro = repro.get("gpu", {})
    software = repro.get("software", {})
    flags = repro.get("torch_flags", {})
    logger.info(
        f"[Repro] driver={gpu_repro.get('driver_version')} "
        f"cuda_built={software.get('torch_cuda_built')} "
        f"cudnn={software.get('cudnn')} "
        f"gpus={gpu_names} "
        f"tf32_matmul={flags.get('allow_tf32_matmul')} "
        f"cudnn_deterministic={flags.get('cudnn_deterministic')} "
        f"cublas={repro.get('env', {}).get('CUBLAS_WORKSPACE_CONFIG')}"
    )
    git = provenance["git"]
    if git.get("available"):
        dirty = " dirty" if git.get("dirty") else ""
        logger.info(f"[Provenance] git {git.get('branch')} {git.get('commit')}{dirty}")
    else:
        logger.info("[Provenance] git unavailable")

    config = {
        "model": "BaselineARAttention",
        "data_dir": str(data_dir),
        "jsonl": jsonl_path.name,
        "T": T,
        "N": N,
        "vocab_size": vocab_size,
        "vocab_size_ext": raw.vocab_size,
        "bos_id": raw.bos_id,
        "eos_id": raw.eos_id,
        "n_train": len(train_list),
        "n_val": len(val_list),
        "d_x": d_x,
        "d_t": d_t,
        "d": d,
        "d_ff": 4 * d,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "dropout": args.dropout,
        "padding_idx": args.padding_idx,
        "train_mode": "ar_fair",
        "epochs": epochs,
        "batch_size": batch_size,
        "loader_batch": loader_batch,
        "effective_batch": loader_batch,
        "lr": lr,
        "optimizer": args.optimizer,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "n_params": n_params,
        "gpu_count": n_gpus,
        "gpu_names": gpu_names,
        "device": str(device),
        "save_checkpoints": args.save_checkpoints,
        "stop_at_val_minimum": args.stop_at_val_minimum,
        "val_minimum_patience": args.val_minimum_patience,
        "val_minimum_ratio": args.val_minimum_ratio,
        "checkpoint_every": args.checkpoint_every,
        "run_tag": tag,
        "provenance": provenance,
    }

    train_losses = []
    val_losses = []
    val_token_accs = []
    val_s_t_ces = []
    epoch_rows = []
    best_val_loss = float("inf")
    best_val_epoch = 0
    start_epoch = 1

    if args.resume_from_checkpoint:
        ckpt_path = find_resume_checkpoint(output_dir, tag)
        if ckpt_path is None:
            logger.info("No checkpoint in output_dir, starting from scratch")
        else:
            logger.info(f"Resuming from checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            unwrap(model).load_state_dict(ckpt["model_state_dict"])
            updater.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            train_losses = list(ckpt.get("train_losses", []))
            val_losses = list(ckpt.get("val_losses", []))
            val_token_accs = list(ckpt.get("val_token_accs", []))
            val_s_t_ces = list(ckpt.get("val_s_t_ces", []))
            best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
            best_val_epoch = int(ckpt.get("best_val_epoch", 0))
            logger.info(
                f"Resumed from epoch {start_epoch - 1}; "
                f"best val {best_val_loss:.6f} at epoch {best_val_epoch}"
            )
            for i, (tr, va) in enumerate(zip(train_losses, val_losses), start=1):
                rec = {
                    "epoch": i,
                    "train_loss": tr,
                    "val_loss": va,
                    "train_mode": "ar_fair",
                }
                if i <= len(val_s_t_ces):
                    rec["val_s_t_ce"] = val_s_t_ces[i - 1]
                if i <= len(val_token_accs):
                    rec["val_token_acc"] = val_token_accs[i - 1]
                epoch_rows.append(rec)

    epoch_jsonl_path = output_dir / f"{tag}.jsonl"
    json_path = output_dir / f"{tag}.json"
    stopped_reason = "completed"

    def snapshot(epoch_done: int):
        elapsed = time.time() - t0
        return {
            "experiment_config": config,
            "training_metrics": {
                "train_losses": train_losses,
                "val_losses": val_losses,
                "val_token_accs": val_token_accs,
                "val_s_t_ces": val_s_t_ces,
                "train_ppl": [_ce_ppl(x) for x in train_losses],
                "val_ppl": [_ce_ppl(x) for x in val_losses],
                "val_s_t_ppl": [_ce_ppl(x) for x in val_s_t_ces],
                "best_val_loss": None if best_val_epoch == 0 else best_val_loss,
                "best_val_epoch": best_val_epoch,
                "last_epoch": epoch_done,
                "stopped_reason": stopped_reason,
                "training_time_sec": elapsed,
                "started_utc": started.isoformat(),
                "val_loss_is": "ar_fair_ce_own_tokens",
                "val_s_t_ce_is": "ar_fair_mean_ce_nonpad_content_ST",
            },
            "provenance": provenance,
        }

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for token_ids_full in train_loader:
            token_ids_full = token_ids_full.to(device, non_blocking=pin)
            loss, _ = updater.step(token_ids_full)
            total_loss += float(loss.item())
            n_batches += 1
        avg_train = total_loss / max(n_batches, 1.0)
        val = evaluate(model, val_loader, device)
        avg_val = val["fair"]
        gap = avg_val - avg_train
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        val_token_accs.append(val["token_acc"])
        val_s_t_ces.append(val["s_t_ce"])
        epoch_rows.append({
            "epoch": len(train_losses),
            "train_loss": avg_train,
            "val_loss": avg_val,
            "val_s_t_ce": val["s_t_ce"],
            "val_token_acc": val["token_acc"],
            "train_mode": "ar_fair",
        })

        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            best_val_epoch = epoch
            logger.info(
                f"NEW BEST VALIDATION FAIR LOSS: {best_val_loss:.6f} at epoch {best_val_epoch}"
            )
            if args.save_checkpoints:
                save_checkpoint(
                    output_dir / f"best_{tag}.pt", model, updater, epoch,
                    avg_train, avg_val, train_losses, val_losses,
                    best_val_loss, best_val_epoch, config,
                    val_token_accs=val_token_accs, val_s_t_ces=val_s_t_ces,
                )

        logger.info(
            f"Epoch [{epoch}/{epochs}] Train AR: {avg_train:.4f} | "
            f"Val AR: {avg_val:.4f} | "
            f"Val S(T) CE: {val['s_t_ce']:.4f} | "
            f"Val S(T) token-acc: {val['token_acc']:.4f} | "
            f"n_tok={val['s_t_n_tok']} | "
            f"gap={gap:+.4f} | Best AR: {best_val_loss:.4f} @ {best_val_epoch}"
        )

        write_losses_jsonl(
            epoch_jsonl_path, epoch_rows,
            header={"experiment_config": config, "provenance": provenance},
        )
        write_metrics_json(json_path, snapshot(epoch))

        if (
            args.save_checkpoints
            and args.checkpoint_every > 0
            and epoch % args.checkpoint_every == 0
        ):
            save_checkpoint(
                output_dir / f"epoch{epoch}_{tag}.pt",
                model, updater, epoch, avg_train, avg_val,
                train_losses, val_losses, best_val_loss, best_val_epoch, config,
                val_token_accs=val_token_accs, val_s_t_ces=val_s_t_ces,
            )

        if args.stop_at_val_minimum and best_val_epoch > 0:
            waited = epoch - best_val_epoch
            if waited >= args.val_minimum_patience and avg_val > best_val_loss * args.val_minimum_ratio:
                stopped_reason = (
                    f"val_minimum: best {best_val_loss:.6f} at epoch {best_val_epoch}, "
                    f"now {avg_val:.6f} after {waited} epochs"
                )
                logger.info(f"VALIDATION MINIMUM REACHED: {stopped_reason}")
                break

    if args.save_checkpoints and train_losses:
        save_checkpoint(
            output_dir / f"final_{tag}.pt", model, updater, len(train_losses),
            train_losses[-1], val_losses[-1], train_losses, val_losses,
            best_val_loss, best_val_epoch, config,
            val_token_accs=val_token_accs, val_s_t_ces=val_s_t_ces,
        )

    payload = snapshot(len(train_losses) if train_losses else 0)
    write_metrics_json(json_path, payload)
    logger.info(
        f"Training finished ({stopped_reason}) in "
        f"{payload['training_metrics']['training_time_sec']:.1f}s"
    )
    if best_val_epoch:
        logger.info(
            f"Best Validation AR Loss: {best_val_loss:.6f} at epoch {best_val_epoch}"
        )
    logger.info(f"Metrics JSON: {json_path}")
    logger.info(f"Epoch JSONL:  {epoch_jsonl_path}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Standard AR QKV baseline train on the same token-window pool. "
        "All values come from the launch *.sh files; this script has no defaults."
    )
    ap.add_argument("--data_dir", type=Path, required=True, help="psf_pool_*.jsonl 이 있는 폴더")
    ap.add_argument("--output_dir", type=Path, required=True, help="folder for run artifacts")
    ap.add_argument("--run_tag", type=str, required=True)
    ap.add_argument("--d_x", type=int, required=True)
    ap.add_argument("--d_t", type=int, required=True)
    ap.add_argument("--n_layers", type=int, required=True)
    ap.add_argument("--n_heads", type=int, required=True)
    ap.add_argument("--dropout", type=float, required=True)
    ap.add_argument("--padding_idx", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch_size", type=int, required=True, help="GPU 당 배치")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--optimizer", type=str, required=True, help="adamw 또는 sgd")
    ap.add_argument("--num_workers", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--deterministic", type=_bool_arg, required=True)
    ap.add_argument("--save_checkpoints", type=_bool_arg, required=True)
    ap.add_argument("--resume_from_checkpoint", type=_bool_arg, required=True)
    ap.add_argument("--stop_at_val_minimum", type=_bool_arg, required=True)
    ap.add_argument("--val_minimum_patience", type=int, required=True)
    ap.add_argument("--val_minimum_ratio", type=float, required=True)
    ap.add_argument(
        "--checkpoint_every", type=int, required=True,
        help="Save epochN_<tag>.pt every N epochs; 0 disables periodic saves",
    )
    return ap.parse_args()


if __name__ == "__main__":
    train_routine(parse_args())
