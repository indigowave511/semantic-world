import argparse
import getpass
import hashlib
import json
import logging
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Must be set before CUDA context creation (matches hook_1130 launch env).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader, Dataset

from four_tensor_model import (
    FourTensorModel,
    FourTensorUpdater,
    build_targets,
    cross_entropy_loss,
    last_slice_token_stats,
)

logger = logging.getLogger("four_tensor_train")

warnings.filterwarnings(
    "ignore",
    message=r"Deterministic behavior was enabled.*uses CuBLAS and you have CUDA",
    category=UserWarning,
)


def setup_logging(*log_paths: Path):
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
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
        logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)


def load_pool_files(data_dir: Path):
    jsonl_paths = sorted(data_dir.glob("psf_pool_*.jsonl"))
    if not jsonl_paths:
        raise FileNotFoundError(f"psf_pool_*.jsonl 이 없습니다: {data_dir}")
    jsonl_path = jsonl_paths[0]
    stem = jsonl_path.name[: -len(".jsonl")]
    meta_path = jsonl_path.with_name(f"{stem}.meta.json")
    vocab_path = jsonl_path.with_name(f"{stem}.vocab.json")
    for path in (meta_path, vocab_path):
        if not path.exists():
            raise FileNotFoundError(f"필수 파일이 없습니다: {path}")
    return jsonl_path, meta_path, vocab_path


def load_meta(meta_path: Path):
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_and_split_data(jsonl_path: Path):
    """
    JSONL 풀을 읽어 train+dev 는 학습용, test 는 검증용으로 나눈다.
    """
    train_data = []
    val_data = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["split"] in ("train", "dev"):
                train_data.append(record)
            elif record["split"] == "test":
                val_data.append(record)

    return train_data, val_data


class ProParaDataset(Dataset):
    """
    states : [T+1, N] == token_ids_full {S(0), ..., S(T)}
    training_step 이 앞 T 개를 입력, 뒤 T 개 ( +1 shift ) 를 타깃으로 쓴다.
    """

    def __init__(self, data_list, T: int, N: int):
        self.data_list = data_list
        self.T = T
        self.N = N

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        record = self.data_list[idx]
        states = torch.tensor(record["states"], dtype=torch.long)
        if states.shape != (self.T + 1, self.N):
            raise ValueError(
                f"states shape {tuple(states.shape)} != ({self.T + 1}, {self.N}) "
                f"(para_id={record.get('para_id')})"
            )
        return states


def optimize_for_rtx3080():
    """Optimize PyTorch settings for dual RTX 3080 (Ubuntu)."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    logger.info("✓ Enabled CUDA optimizations for RTX 3080")


def set_seed_for_reproducibility(seed: int, deterministic: bool = True):
    """Set all random seeds for reproducibility across experiments and machines."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=False)
        except AttributeError:
            pass
        except RuntimeError as e:
            logger.warning(f"Could not enable full deterministic mode: {e}")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
        logger.info(f"✓ Set seed to {seed} with deterministic algorithms enabled")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        logger.info(f"✓ Set seed to {seed} (non-deterministic mode for performance)")

    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"✓ All random seeds set to {seed} for reproducibility")


def setup_dual_gpu_model(model):
    """Setup model for dual-GPU DataParallel, matching hook_1130 on this Ubuntu box."""
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available; training on CPU")
        device = torch.device("cpu")
        return model.to(device), device

    device = torch.device("cuda")
    n_gpus = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
    if n_gpus > 1:
        logger.info(f"Using {n_gpus} GPUs: {names}")
        model = DataParallel(model)
    else:
        logger.info(f"Using single GPU: {names[0]}")
    return model.to(device), device


def unwrap(model):
    return model.module if isinstance(model, DataParallel) else model


def n_parallel_devices(model) -> int:
    if isinstance(model, DataParallel):
        return max(len(model.device_ids), 1)
    return 1


def model_forward(model, token_ids: torch.Tensor) -> torch.Tensor:
    """DataParallel needs batch dim >= n_gpus; fall back to the raw module otherwise."""
    if isinstance(model, DataParallel) and token_ids.size(0) < n_parallel_devices(model):
        return unwrap(model)(token_ids)
    return model(token_ids)


@torch.no_grad()
def evaluate(model, loader, device):
    """window CE is the train val_loss. S(T) CE/acc are the comparison columns."""
    model.eval()
    raw = unwrap(model)
    pad = raw.padding_idx
    pin = device.type == "cuda"
    window_total = 0.0
    n_batches = 0
    st_ce = 0.0
    st_correct = 0
    st_n = 0
    for token_ids_full in loader:
        token_ids_full = token_ids_full.to(device, non_blocking=pin)
        logits = model_forward(model, token_ids_full[:, :-1])
        Y = build_targets(token_ids_full)
        loss = cross_entropy_loss(logits, Y, pad_id=pad)
        window_total += loss.item()
        n_batches += 1
        ce_sum, n_ok, n_tok = last_slice_token_stats(logits, token_ids_full, pad)
        st_ce += float(ce_sum.item())
        st_correct += int(n_ok.item())
        st_n += int(n_tok.item())
    st_n = max(st_n, 1)
    return {
        "window": window_total / max(n_batches, 1.0),
        "s_t_ce": st_ce / st_n,
        "s_t_acc": st_correct / st_n,
        "s_t_n_tok": st_n,
        "s_t_n_matched": st_correct,
        "s_t_ce_sum": st_ce,
    }


def _loader_kwargs(num_workers: int, pin_memory: bool, seed: int, shuffle: bool):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True

        def worker_init_fn(worker_id):
            worker_seed = seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        kwargs["worker_init_fn"] = worker_init_fn
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
        kwargs["generator"] = generator
    return kwargs


def _ce_ppl(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def _utc_from_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _source_file_note(path: Path) -> dict:
    path = path.resolve()
    st = path.stat()
    note = {
        "path": str(path),
        "size_bytes": st.st_size,
        "mtime_utc": _utc_from_ts(st.st_mtime),
        "ctime_utc": _utc_from_ts(st.st_ctime),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    birth = getattr(st, "st_birthtime", None)
    if birth is not None:
        note["created_utc"] = _utc_from_ts(birth)
    return note


def _run_capture(cmd) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=8)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    out = (r.stdout or "").strip()
    return out or None


def _git_note(repo: Path) -> dict:
    def git(*args):
        return _run_capture(["git", "-C", str(repo), *args])

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"available": False}
    dirty = git("status", "--porcelain")
    return {
        "available": True,
        "commit": commit,
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty),
    }


def _nvidia_proc_version() -> str | None:
    path = Path("/proc/driver/nvidia/version")
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip() or None
    except OSError:
        return None


def _nvidia_smi_gpus() -> list:
    raw = _run_capture([
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if not raw:
        return []
    gpus = []
    for line in raw.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        rec = {
            "index": int(parts[0]) if parts[0].isdigit() else parts[0],
            "name": parts[1],
            "uuid": parts[2],
            "driver_version": parts[3],
            "memory_total_mib": int(parts[4]) if parts[4].isdigit() else parts[4],
            "compute_cap": parts[5],
        }
        gpus.append(rec)
    return gpus


def _nvidia_driver_version(smi_gpus: list) -> str | None:
    if smi_gpus:
        return smi_gpus[0].get("driver_version")
    raw = _run_capture(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if raw:
        return raw.splitlines()[0].strip()
    proc = _nvidia_proc_version()
    if not proc:
        return None
    for token in proc.replace("\n", " ").split():
        if token[0].isdigit() and "." in token:
            return token
    return None


def _repro_env() -> dict:
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "CUBLAS_WORKSPACE_CONFIG",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TOKENIZERS_PARALLELISM",
        "PYTHONHASHSEED",
        "NVIDIA_VISIBLE_DEVICES",
        "CUDA_LAUNCH_BLOCKING",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    )
    return {k: os.environ.get(k) for k in keys if os.environ.get(k) is not None}


def _software_versions() -> dict:
    note = {
        "python": sys.version.split()[0],
        "python_full": sys.version,
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_built": torch.version.cuda,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "debug_build": bool(getattr(torch.version, "debug", False)),
    }
    try:
        note["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        note["cudnn"] = None
    try:
        note["nccl"] = torch.cuda.nccl.version() if torch.cuda.is_available() else None
    except Exception:
        note["nccl"] = None
    return note


def collect_provenance() -> dict:
    here = Path(__file__).resolve()
    model_src = here.with_name("four_tensor_model.py")
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
            "four_tensor_model.py": _source_file_note(model_src),
            "four_tensor_train.py": _source_file_note(here),
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


def fill_runtime_reproducibility(provenance: dict, args) -> dict:
    """Capture torch/CUDA flags after seed and GPU init so they match the actual run."""
    repro = provenance.setdefault("reproducibility", {})
    repro["seed"] = args.seed
    repro["deterministic_requested"] = args.deterministic
    repro["pythonhashseed"] = os.environ.get("PYTHONHASHSEED")
    repro["torch_num_threads"] = torch.get_num_threads()
    try:
        repro["deterministic_algorithms_enabled"] = torch.are_deterministic_algorithms_enabled()
    except Exception:
        repro["deterministic_algorithms_enabled"] = None

    flags = {
        "cudnn_enabled": torch.backends.cudnn.enabled,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    try:
        flags["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    except Exception:
        flags["float32_matmul_precision"] = None
    repro["torch_flags"] = flags

    gpu = repro.setdefault("gpu", {})
    gpu["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        gpu["device_count"] = torch.cuda.device_count()
        gpu["current_device"] = torch.cuda.current_device()
        devices = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            devices.append({
                "index": i,
                "name": p.name,
                "capability": f"{p.major}.{p.minor}",
                "total_memory_bytes": p.total_memory,
                "multi_processor_count": p.multi_processor_count,
            })
        gpu["torch_devices"] = devices
        try:
            gpu["cuda_runtime_version"] = torch.version.cuda
        except Exception:
            gpu["cuda_runtime_version"] = None
    else:
        gpu["device_count"] = 0
        gpu["torch_devices"] = []

    software = repro.setdefault("software", {})
    try:
        software["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            software["nccl"] = torch.cuda.nccl.version()
    except Exception:
        pass
    return repro


def _bool_arg(value: str) -> bool:
    key = value.strip().lower()
    if key in ("1", "true", "yes", "y"):
        return True
    if key in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")


def find_resume_checkpoint(output_dir: Path, tag: str):
    candidates = []
    for name in (f"best_{tag}.pt", f"final_{tag}.pt"):
        path = output_dir / name
        if path.exists():
            candidates.append(path)
    candidates.extend(sorted(output_dir.glob(f"epoch*_{tag}.pt")))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def save_checkpoint(path: Path, model, updater, epoch, avg_train, avg_val,
                    train_losses, val_losses, best_val_loss, best_val_epoch, config,
                    val_s_t_ces=None, val_s_t_accs=None):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": unwrap(model).state_dict(),
            "optimizer_state_dict": updater.optimizer.state_dict(),
            "train_loss": avg_train,
            "val_loss": avg_val,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "val_s_t_ces": [] if val_s_t_ces is None else val_s_t_ces,
            "val_s_t_accs": [] if val_s_t_accs is None else val_s_t_accs,
            "best_val_loss": best_val_loss,
            "best_val_epoch": best_val_epoch,
            "config": config,
        },
        path,
    )


def write_metrics_json(path: Path, payload: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_losses_jsonl(
    path: Path, train_losses, val_losses, header: dict,
    val_s_t_ces=None, val_s_t_accs=None,
):
    running_best = float("inf")
    running_best_epoch = 0
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"record": "run_header", **header}) + "\n")
        for i, (tr, va) in enumerate(zip(train_losses, val_losses), start=1):
            if va < running_best:
                running_best = va
                running_best_epoch = i
            rec = {
                "record": "epoch",
                "epoch": i,
                "train_loss": tr,
                "val_loss": va,
                "train_ppl": _ce_ppl(tr),
                "val_ppl": _ce_ppl(va),
                "gap": va - tr,
                "best_val_loss": running_best,
                "best_val_epoch": running_best_epoch,
            }
            if val_s_t_ces is not None and i <= len(val_s_t_ces):
                rec["val_s_t_ce"] = val_s_t_ces[i - 1]
                rec["val_s_t_ppl"] = _ce_ppl(val_s_t_ces[i - 1])
            if val_s_t_accs is not None and i <= len(val_s_t_accs):
                rec["val_s_t_token_acc"] = val_s_t_accs[i - 1]
            f.write(json.dumps(rec) + "\n")


def train_routine(args):
    output_dir = Path(args.output_dir)
    tag = args.run_tag
    cwd_log = Path.cwd() / "four_tensor_train.log"
    run_log = output_dir / f"{tag}.log"
    # Shell launch tees the whole terminal into four_tensor_train.log;
    # writing it again here would truncate that transcript and duplicate lines.
    if os.environ.get("FOUR_TENSOR_TERMINAL_LOG"):
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

    model = FourTensorModel(
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

    updater = FourTensorUpdater(model, lr=lr, optimizer=args.optimizer)
    n_params = sum(p.numel() for p in unwrap(model).parameters() if p.requires_grad)
    logger.info(
        f"[Model] FourTensorModel on {device}  gpus={n_gpus}  params={n_params:,}"
    )
    logger.info(
        f"[Model] d_x={d_x} d_t={d_t} n_layers={n_layers} n_heads={n_heads} "
        f"dropout={args.dropout} padding_idx={args.padding_idx} "
        f"batch/gpu={batch_size} loader_batch={loader_batch} "
        f"effective_batch={loader_batch}"
    )
    logger.info(f"Random seed: {args.seed} (deterministic mode: {args.deterministic})")
    provenance["machine"]["cuda_available"] = torch.cuda.is_available()
    provenance["machine"]["gpu_names"] = gpu_names
    repro = fill_runtime_reproducibility(provenance, args)
    src = provenance["source_files"]["four_tensor_model.py"]
    logger.info(
        f"[Provenance] four_tensor_model.py mtime={src['mtime_utc']} "
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
        "data_dir": str(data_dir),
        "jsonl": jsonl_path.name,
        "T": T,
        "N": N,
        "vocab_size": vocab_size,
        "n_train": len(train_list),
        "n_val": len(val_list),
        "d_x": d_x,
        "d_t": d_t,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "dropout": args.dropout,
        "padding_idx": args.padding_idx,
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
    val_s_t_ces = []
    val_s_t_accs = []
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
            val_s_t_ces = list(ckpt.get("val_s_t_ces", []))
            val_s_t_accs = list(ckpt.get("val_s_t_accs", []))
            best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
            best_val_epoch = int(ckpt.get("best_val_epoch", 0))
            logger.info(
                f"Resumed from epoch {start_epoch - 1}; "
                f"best val {best_val_loss:.6f} at epoch {best_val_epoch}"
            )

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
                "val_s_t_ces": val_s_t_ces,
                "val_s_t_accs": val_s_t_accs,
                "train_ppl": [_ce_ppl(x) for x in train_losses],
                "val_ppl": [_ce_ppl(x) for x in val_losses],
                "val_s_t_ppl": [_ce_ppl(x) for x in val_s_t_ces],
                "best_val_loss": None if best_val_epoch == 0 else best_val_loss,
                "best_val_epoch": best_val_epoch,
                "last_epoch": epoch_done,
                "stopped_reason": stopped_reason,
                "training_time_sec": elapsed,
                "started_utc": started.isoformat(),
                "val_loss_is": "window_ce_S1_to_ST",
                "val_s_t_ce_is": "last_slice_mean_ce_nonpad_ST",
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
            total_loss += loss.item()
            n_batches += 1
        avg_train = total_loss / max(n_batches, 1.0)
        val = evaluate(model, val_loader, device)
        avg_val = val["window"]
        gap = avg_val - avg_train
        train_losses.append(avg_train)
        val_losses.append(avg_val)
        val_s_t_ces.append(val["s_t_ce"])
        val_s_t_accs.append(val["s_t_acc"])

        is_best = avg_val < best_val_loss
        if is_best:
            best_val_loss = avg_val
            best_val_epoch = epoch
            logger.info(
                f"NEW BEST VALIDATION LOSS: {best_val_loss:.6f} at epoch {best_val_epoch}"
            )
            if args.save_checkpoints:
                save_checkpoint(
                    output_dir / f"best_{tag}.pt", model, updater, epoch,
                    avg_train, avg_val, train_losses, val_losses,
                    best_val_loss, best_val_epoch, config,
                    val_s_t_ces=val_s_t_ces, val_s_t_accs=val_s_t_accs,
                )

        logger.info(
            f"Epoch [{epoch}/{epochs}] Train Loss: {avg_train:.4f} | "
            f"Val Loss: {avg_val:.4f} | "
            f"Val S(T) CE: {val['s_t_ce']:.4f} | "
            f"Val S(T) token-acc: {val['s_t_acc']:.4f} | "
            f"n_tok={val['s_t_n_tok']} | "
            f"gap={gap:+.4f} | Best Val: {best_val_loss:.4f} @ {best_val_epoch}"
        )

        write_losses_jsonl(
            epoch_jsonl_path, train_losses, val_losses,
            header={"experiment_config": config, "provenance": provenance},
            val_s_t_ces=val_s_t_ces, val_s_t_accs=val_s_t_accs,
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
                val_s_t_ces=val_s_t_ces, val_s_t_accs=val_s_t_accs,
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
            val_s_t_ces=val_s_t_ces, val_s_t_accs=val_s_t_accs,
        )

    payload = snapshot(len(train_losses) if train_losses else 0)
    write_metrics_json(json_path, payload)
    logger.info(
        f"Training finished ({stopped_reason}) in "
        f"{payload['training_metrics']['training_time_sec']:.1f}s"
    )
    if best_val_epoch:
        logger.info(
            f"Best Validation Loss: {best_val_loss:.6f} at epoch {best_val_epoch}"
        )
    logger.info(f"Metrics JSON: {json_path}")
    logger.info(f"Epoch JSONL:  {epoch_jsonl_path}")


def parse_args():
    ap = argparse.ArgumentParser(
        description="PSF 4-tensor train on a token-window pool (DataParallel, dual RTX 3080). "
        "All values come from the launch *.sh files; this script has no defaults."
    )
    ap.add_argument("--data_dir", type=Path, required=True, help="psf_pool_*.jsonl 이 있는 폴더")
    ap.add_argument("--output_dir", type=Path, required=True, help="folder for run artifacts")
    ap.add_argument("--run_tag", type=str, required=True, help="shared filename tag, e.g. rocstories_T4N16_...")
    ap.add_argument("--d_x", type=int, required=True)
    ap.add_argument("--d_t", type=int, required=True)
    ap.add_argument("--n_layers", type=int, required=True)
    ap.add_argument("--n_heads", type=int, required=True)
    ap.add_argument("--dropout", type=float, required=True, help="residual-branch dropout p")
    ap.add_argument("--padding_idx", type=int, required=True, help="vocab id of <PAD>")
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
