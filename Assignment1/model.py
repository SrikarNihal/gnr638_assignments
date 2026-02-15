"""Minimal CNN training using the C++ Tensor/autograd backend.

- Optimizer: Adam (C++ / dylib)
- Loss: categorical negative log-likelihood (NLL)

Datasets supported:
- Numeric folders: data_1/{0..9}/* (images)
- Named folders:   data_2/<class_name>/* (images)

Environment variables (all optional):

Outputs:
- WEIGHTS_OUT: output checkpoint path (default: cnn_weights.pkl)
- CLASSES_OUT: output class-name mapping (default: class_names.json)

Training core:
- EPOCHS: training epochs (default: 8)
- BATCH_SIZE: mini-batch size (default: 64)
- LR: learning rate (default: 1e-3)
- SEED: RNG seed used for shuffle+augmentation (default: 0)

Optimizer (Adam):
- ADAM_BETA1: default 0.9
- ADAM_BETA2: default 0.999
- ADAM_EPS: default 1e-8
- WEIGHT_DECAY: default 0.0

Logging / validation cadence:
- LOG_EVERY_SEC: seconds between progress logs (default: 30; 0 disables time-based logging)
- LOG_EVERY_BATCHES: if >0, print batch loss/acc every N batches (default: 0)
- METRICS_EVERY_BATCHES: if >0, print batch_loss/batch_acc every N batches (default: 0)
- VAL_EVERY: validate every N epochs (default: 1). Last epoch always validates.


Augmentation:
- AUGMENT: 0/1 or true/false (default: 0)
- AUG_FLIP_PROB: default 0.5
- AUG_TRANSLATE: max translate fraction of width/height (default: 0.1)
- AUG_ROTATE_DEG: max rotation degrees (default: 5)

Model hparams:
- BN_MOMENTUM: default 0.1
- BN_EPS: default 1e-5
- DROPOUT_CONV: default 0.25
- DROPOUT_DENSE: default 0.5
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import pickle
import random
import time
from pathlib import Path
from typing import List, Tuple

import cv2

from tensor_ctypes import Adam, Tensor


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return int(default)
    try:
        return int(v)
    except ValueError as e:
        raise ValueError(f"env var {name} must be an int, got: {v!r}") from e


def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None or v == "":
        return float(default)
    try:
        return float(v)
    except ValueError as e:
        raise ValueError(f"env var {name} must be a float, got: {v!r}") from e


def _env_str(name: str, default: str) -> str:
    v = os.environ.get(name)
    if v is None or v == "":
        return str(default)
    return str(v)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return bool(default)
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"env var {name} must be bool-like (0/1/true/false), got: {v!r}")


### Dataset and image loading
#
# Dataset scanning and image decoding live in the notebook (test.ipynb) by design.
# The training loop consumes:
# - train_items / val_items: list[(Path, int)]
# - load_arr(path): -> 32x32 RGB uint8 HWC (OpenCV-like)


def _rgb_u8_hwc_to_chw01_flat(rgb) -> list[float]:
    """Convert 32x32 RGB uint8 HWC image to CHW float list in [0,1]."""
    # Use Python indexing (works on OpenCV image arrays).
    out: list[float] = [0.0] * (3 * 32 * 32)
    for c in range(3):
        base_c = c * 32 * 32
        for y in range(32):
            row = rgb[y]
            base_y = base_c + y * 32
            for x in range(32):
                out[base_y + x] = float(row[x][c]) / 255.0
    return out

def _augment_rgb_u8_hwc_32x32(
    rgb,
    rng: random.Random,
    *,
    hflip_prob: float = 0.5,
    max_translate_frac: float = 0.1,
    max_rotate_deg: float = 5.0,
):
    """Simple augmentation for 32x32 RGB uint8 HWC image.

    Uses OpenCV ops only.
    """
    if rgb is None:
        return rgb

    if hflip_prob > 0 and rng.random() < hflip_prob:
        rgb = cv2.flip(rgb, 1)

    do_affine = (max_translate_frac and max_translate_frac > 0) or (max_rotate_deg and max_rotate_deg > 0)
    if do_affine:
        angle = (rng.random() * 2.0 - 1.0) * float(max_rotate_deg)
        tx = (rng.random() * 2.0 - 1.0) * float(max_translate_frac) * 32.0
        ty = (rng.random() * 2.0 - 1.0) * float(max_translate_frac) * 32.0
        M = cv2.getRotationMatrix2D((16.0, 16.0), float(angle), 1.0)
        M[0, 2] += float(tx)
        M[1, 2] += float(ty)
        rgb = cv2.warpAffine(
            rgb,
            M,
            (32, 32),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        # warpAffine can return HxW for single-channel; enforce HWC
        if getattr(rgb, "ndim", 0) == 2:
            rgb = rgb[:, :, None]

    return rgb

def _he_std(fan_in: int) -> float:
    return (2.0 / float(fan_in)) ** 0.5


@dataclass
class Params:
    t: Tensor
    v: Tensor


class CNN:
    def __init__(self, num_classes: int = 10):
        if int(num_classes) < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        self.num_classes = int(num_classes)
        # Conv weights: (out_c, in_c, k, k), bias: (out_c,)
        self.w1 = Tensor.randn((32, 3, 3, 3), stdev=_he_std(3 * 3 * 3), requires_grad=True)
        self.b1 = Tensor.zeros((32,), requires_grad=True)

        self.w2 = Tensor.randn((32, 32, 3, 3), stdev=_he_std(32 * 3 * 3), requires_grad=True)
        self.b2 = Tensor.zeros((32,), requires_grad=True)

        self.w3 = Tensor.randn((64, 32, 3, 3), stdev=_he_std(32 * 3 * 3), requires_grad=True)
        self.b3 = Tensor.zeros((64,), requires_grad=True)

        self.w4 = Tensor.randn((64, 64, 3, 3), stdev=_he_std(64 * 3 * 3), requires_grad=True)
        self.b4 = Tensor.zeros((64,), requires_grad=True)

        # Match the requested Keras architecture:
        # Input 32x32 -> valid conv: 30 -> valid conv: 28 -> pool2: 14
        # valid conv: 12 -> valid conv: 10 -> pool2: 5
        # Flatten: 64 * 5 * 5 = 1600
        self.fc1_w = Tensor.randn((1600, 512), stdev=_he_std(1600), requires_grad=True)
        self.fc1_b = Tensor.zeros((512,), requires_grad=True)

        self.fc2_w = Tensor.randn((512, self.num_classes), stdev=_he_std(512), requires_grad=True)
        self.fc2_b = Tensor.zeros((self.num_classes,), requires_grad=True)

        # BatchNorm params (trainable) + running stats (buffers, no grad)
        def _bn(ch: int, name: str):
            g = Tensor.ones((ch,), requires_grad=True)
            be = Tensor.zeros((ch,), requires_grad=True)
            rm = Tensor.zeros((ch,), requires_grad=False)
            rv = Tensor.ones((ch,), requires_grad=False)
            setattr(self, f"{name}_g", g)
            setattr(self, f"{name}_b", be)
            setattr(self, f"{name}_rm", rm)
            setattr(self, f"{name}_rv", rv)

        _bn(32, "bn1")
        _bn(32, "bn2")
        _bn(64, "bn3")
        _bn(64, "bn4")

        self.bn_momentum = _env_float("BN_MOMENTUM", 0.1)
        self.bn_eps = _env_float("BN_EPS", 1e-5)
        self.dropout_p_conv = _env_float("DROPOUT_CONV", 0.25)
        self.dropout_p_dense = _env_float("DROPOUT_DENSE", 0.5)

    def parameters(self) -> List[Tensor]:
        return [
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            self.w3,
            self.b3,
            self.w4,
            self.b4,
            self.fc1_w,
            self.fc1_b,
            self.fc2_w,
            self.fc2_b,
            self.bn1_g,
            self.bn1_b,
            self.bn2_g,
            self.bn2_b,
            self.bn3_g,
            self.bn3_b,
            self.bn4_g,
            self.bn4_b,
        ]

    def buffers(self) -> List[Tensor]:
        return [
            self.bn1_rm,
            self.bn1_rv,
            self.bn2_rm,
            self.bn2_rv,
            self.bn3_rm,
            self.bn3_rv,
            self.bn4_rm,
            self.bn4_rv,
        ]

    def named_parameters(self) -> List[Tuple[str, Tensor]]:
        return [
            ("w1", self.w1),
            ("b1", self.b1),
            ("w2", self.w2),
            ("b2", self.b2),
            ("w3", self.w3),
            ("b3", self.b3),
            ("w4", self.w4),
            ("b4", self.b4),
            ("fc1_w", self.fc1_w),
            ("fc1_b", self.fc1_b),
            ("fc2_w", self.fc2_w),
            ("fc2_b", self.fc2_b),
            ("bn1_g", self.bn1_g),
            ("bn1_b", self.bn1_b),
            ("bn2_g", self.bn2_g),
            ("bn2_b", self.bn2_b),
            ("bn3_g", self.bn3_g),
            ("bn3_b", self.bn3_b),
            ("bn4_g", self.bn4_g),
            ("bn4_b", self.bn4_b),
        ]

    def named_tensors(self) -> List[Tuple[str, Tensor]]:
        return self.named_parameters() + [
            ("bn1_rm", self.bn1_rm),
            ("bn1_rv", self.bn1_rv),
            ("bn2_rm", self.bn2_rm),
            ("bn2_rv", self.bn2_rv),
            ("bn3_rm", self.bn3_rm),
            ("bn3_rv", self.bn3_rv),
            ("bn4_rm", self.bn4_rm),
            ("bn4_rv", self.bn4_rv),
        ]

    def forward_logits(self, x_nchw: Tensor, *, training: bool = True) -> Tensor:
        # Input is NCHW (N,3,32,32)
        x = x_nchw

        # Keras-like Block 1 (Conv+ReLU -> BN -> Conv+ReLU -> BN -> Pool -> Dropout)
        x = x.conv2d(self.w1, self.b1, stride=1, padding=0).relu()
        x = x.batchnorm2d(
            self.bn1_g,
            self.bn1_b,
            self.bn1_rm,
            self.bn1_rv,
            momentum=self.bn_momentum,
            eps=self.bn_eps,
            training=training,
        )
        x = x.conv2d(self.w2, self.b2, stride=1, padding=0).relu()
        x = x.batchnorm2d(
            self.bn2_g,
            self.bn2_b,
            self.bn2_rm,
            self.bn2_rv,
            momentum=self.bn_momentum,
            eps=self.bn_eps,
            training=training,
        )
        x = x.maxpool2d(kernel=2, stride=2)
        x = x.dropout(p=self.dropout_p_conv, training=training)

        # Keras-like Block 2
        x = x.conv2d(self.w3, self.b3, stride=1, padding=0).relu()
        x = x.batchnorm2d(
            self.bn3_g,
            self.bn3_b,
            self.bn3_rm,
            self.bn3_rv,
            momentum=self.bn_momentum,
            eps=self.bn_eps,
            training=training,
        )
        x = x.conv2d(self.w4, self.b4, stride=1, padding=0).relu()
        x = x.batchnorm2d(
            self.bn4_g,
            self.bn4_b,
            self.bn4_rm,
            self.bn4_rv,
            momentum=self.bn_momentum,
            eps=self.bn_eps,
            training=training,
        )
        x = x.maxpool2d(kernel=2, stride=2)
        x = x.dropout(p=self.dropout_p_conv, training=training)

        # Flatten
        s = x.shape()
        n = int(s[0])
        flat = 1
        for v in s[1:]:
            flat *= int(v)
        x = x.reshape((n, flat))

        # Dense 512 + dropout
        x = x.matmul(self.fc1_w).add(self.fc1_b.reshape((1, 512))).relu()
        x = x.dropout(p=self.dropout_p_dense, training=training)

        # Dense classes (logits)
        logits = x.matmul(self.fc2_w).add(self.fc2_b.reshape((1, self.num_classes)))
        return logits

    def forward(self, x_nchw: Tensor, *, training: bool = True) -> Tensor:
        logits = self.forward_logits(x_nchw, training=training)
        return logits.softmax2d()


def _batch_iter(indices: list[int], batch_size: int):
    for i in range(0, len(indices), batch_size):
        yield indices[i : i + batch_size]


def _eval_epoch(
    model: CNN,
    items: list[tuple[Path, int]],
    batch_size: int,
    load_arr,
) -> tuple[float, float]:
    total_loss = 0.0
    total_n = 0
    total_correct = 0

    order = list(range(len(items)))
    for batch_idx in _batch_iter(order, batch_size):
        ys = [int(items[j][1]) for j in batch_idx]
        bsz = len(batch_idx)
        flat = [0.0] * (bsz * 3 * 32 * 32)
        for bi, j in enumerate(batch_idx):
            rgb = load_arr(items[j][0])
            img_flat = _rgb_u8_hwc_to_chw01_flat(rgb)
            off = bi * 3 * 32 * 32
            flat[off : off + 3 * 32 * 32] = img_flat
        x = Tensor.create(flat, (bsz, 3, 32, 32), requires_grad=False)

        logits = model.forward_logits(x, training=False)
        loss = logits.categorical_crossentropy_logits(ys)
        loss_val = float(loss.to_list()[0])

        l = logits.to_list()  # flat (B*C)
        C = int(logits.shape()[1])
        correct = 0
        for i in range(bsz):
            row = l[i * C : (i + 1) * C]
            pred_i = max(range(C), key=row.__getitem__)
            if int(pred_i) == int(ys[i]):
                correct += 1
        total_correct += correct

        total_loss += loss_val * len(batch_idx)
        total_n += len(batch_idx)

    avg_loss = total_loss / float(total_n) if total_n else 0.0
    acc = total_correct / float(total_n) if total_n else 0.0
    return avg_loss, acc


def train(
    model: CNN,
    train_items: list[tuple[Path, int]],
    val_items: list[tuple[Path, int]],
    *,
    batch_size: int = 64,
    epochs: int = 8,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    seed: int = 0,
    load_arr,
    return_history: bool = False,
) -> list[dict] | None:
    opt = Adam()
    params = model.parameters()
    m1: List[Tensor] = [Tensor.zeros(p.shape(), requires_grad=False) for p in params]
    m2: List[Tensor] = [Tensor.zeros(p.shape(), requires_grad=False) for p in params]
    t = 0

    if load_arr is None:
        raise ValueError("train(): load_arr must be provided (dataset/image loading is handled in the notebook)")
    rng = random.Random(int(seed))

    history: list[dict] | None = [] if bool(return_history) else None

    log_every_sec = float(_env_float("LOG_EVERY_SEC", 30.0))
    log_every_batches = int(_env_int("LOG_EVERY_BATCHES", 0))
    metrics_every_batches = int(_env_int("METRICS_EVERY_BATCHES", 0))

    if log_every_sec < 0:
        raise ValueError(f"LOG_EVERY_SEC must be >= 0, got {log_every_sec}")
    if log_every_batches < 0:
        raise ValueError(f"LOG_EVERY_BATCHES must be >= 0, got {log_every_batches}")
    if metrics_every_batches < 0:
        raise ValueError(f"METRICS_EVERY_BATCHES must be >= 0, got {metrics_every_batches}")

    val_every = int(_env_int("VAL_EVERY", 1))
    if val_every < 1:
        raise ValueError(f"VAL_EVERY must be >= 1, got {val_every}")
    cur_lr = float(lr)

    augment = _env_bool("AUGMENT", False)
    aug_flip_prob = _env_float("AUG_FLIP_PROB", 0.5)
    aug_translate = _env_float("AUG_TRANSLATE", 0.1)
    aug_rotate = _env_float("AUG_ROTATE_DEG", 5.0)

    for epoch in range(int(epochs)):
        epoch_t0 = time.perf_counter()
        order = list(range(len(train_items)))
        rng.shuffle(order)

        steps = (len(order) + int(batch_size) - 1) // int(batch_size)
        if steps == 0:
            raise ValueError("empty training set")

        epoch_loss_sum = 0.0
        epoch_loss_n = 0
        epoch_correct = 0
        epoch_acc_n = 0

        last_log_t = time.perf_counter()
        run_loss_sum = 0.0
        run_loss_n = 0
        run_correct = 0
        run_acc_n = 0

        for step_i, batch_idx in enumerate(_batch_iter(order, int(batch_size)), start=1):
            # Heartbeat: print BEFORE the expensive forward/backward.
            # This avoids "no output for minutes" when a single batch is slow.
            if step_i == 1 or (time.perf_counter() - last_log_t) >= float(log_every_sec):
                print(f"epoch {epoch+1}/{epochs}  step {step_i}/{steps}  running...", flush=True)
                last_log_t = time.perf_counter()

            ys = [int(train_items[j][1]) for j in batch_idx]
            bsz = len(batch_idx)

            flat = [0.0] * (bsz * 3 * 32 * 32)
            for bi, j in enumerate(batch_idx):
                rgb = load_arr(train_items[j][0])
                if augment:
                    rgb = _augment_rgb_u8_hwc_32x32(
                        rgb,
                        rng,
                        hflip_prob=float(aug_flip_prob),
                        max_translate_frac=float(aug_translate),
                        max_rotate_deg=float(aug_rotate),
                    )
                img_flat = _rgb_u8_hwc_to_chw01_flat(rgb)
                off = bi * 3 * 32 * 32
                flat[off : off + 3 * 32 * 32] = img_flat
            x = Tensor.create(flat, (bsz, 3, 32, 32), requires_grad=False)

            for p in params:
                p.zero_grad()

            logits = model.forward_logits(x, training=True)
            loss = logits.categorical_crossentropy_logits(ys)
            loss.backward()

            t += 1
            for p, mm, vv in zip(params, m1, m2):
                opt.step(
                    p,
                    mm,
                    vv,
                    lr=float(cur_lr),
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=float(weight_decay),
                    t=t,
                )

            loss_val = float(loss.to_list()[0])
            epoch_loss_sum += loss_val * len(batch_idx)
            epoch_loss_n += len(batch_idx)

            run_loss_sum += loss_val * len(batch_idx)
            run_loss_n += len(batch_idx)

            # Accuracy each batch (keeps epoch accuracy exact).
            l_flat = logits.to_list()  # flat (B*C)
            C = int(logits.shape()[1])
            correct = 0
            for i in range(bsz):
                row = l_flat[i * C : (i + 1) * C]
                pred_i = max(range(C), key=row.__getitem__)
                if int(pred_i) == int(ys[i]):
                    correct += 1
            epoch_correct += correct
            run_correct += correct
            epoch_acc_n += len(batch_idx)
            run_acc_n += len(batch_idx)

            # Optional per-batch metrics
            if metrics_every_batches > 0 and (step_i % metrics_every_batches == 0 or step_i == steps):
                batch_acc = (correct / float(bsz)) if bsz else 0.0
                print(
                    f"epoch {epoch+1}/{epochs}  step {step_i}/{steps}  "
                    f"batch_loss={loss_val:.6f} batch_acc={batch_acc:.4f}",
                    flush=True,
                )

            # Optional periodic logging (time-based or batch-based)
            now = time.perf_counter()
            should_log_batch = log_every_batches > 0 and (step_i % log_every_batches == 0 or step_i == steps)
            should_log_time = log_every_sec > 0 and run_loss_n > 0 and ((now - last_log_t) >= float(log_every_sec) or step_i == steps)

            if should_log_batch and metrics_every_batches <= 0:
                batch_acc = (correct / float(bsz)) if bsz else 0.0
                print(
                    f"epoch {epoch+1}/{epochs}  step {step_i}/{steps}  "
                    f"loss={loss_val:.6f} acc={batch_acc:.4f}",
                    flush=True,
                )

            if should_log_time:
                avg_loss = run_loss_sum / float(run_loss_n) if run_loss_n else 0.0
                acc_str = f"{(run_correct / float(run_acc_n)):.4f}" if run_acc_n else "n/a"
                print(
                    f"epoch {epoch+1}/{epochs}  step {step_i}/{steps}  "
                    f"avg_loss={avg_loss:.6f} avg_acc={acc_str}",
                    flush=True,
                )
                last_log_t = now
                run_loss_sum = 0.0
                run_correct = 0
                run_loss_n = 0
                run_acc_n = 0

        train_loss = epoch_loss_sum / float(epoch_loss_n) if epoch_loss_n else 0.0
        train_acc = epoch_correct / float(epoch_acc_n) if epoch_acc_n else 0.0

        did_val = False
        val_loss = 0.0
        val_acc = 0.0
        do_val_this_epoch = bool(val_items) and (((epoch + 1) % int(val_every)) == 0 or (epoch + 1) == int(epochs))
        if do_val_this_epoch:
            val_loss, val_acc = _eval_epoch(model, val_items, batch_size=max(1, int(batch_size)), load_arr=load_arr)
            did_val = True

        val_loss_str = f"{val_loss:.6f}" if did_val else "n/a"
        val_acc_str = f"{val_acc:.4f}" if did_val else "n/a"

        print(
            f"epoch {epoch+1}/{epochs}  lr={cur_lr:.6g}  "
            f"train_loss={train_loss:.6f} train_acc={train_acc:.4f}  "
            f"val_loss={val_loss_str} val_acc={val_acc_str}"
        )

        if history is not None:
            epoch_t1 = time.perf_counter()
            history.append(
                {
                    "epoch": int(epoch) + 1,
                    "lr": float(cur_lr),
                    "train_loss": float(train_loss),
                    "train_acc": float(train_acc),
                    "val_loss": (float(val_loss) if did_val else None),
                    "val_acc": (float(val_acc) if did_val else None),
                    "seconds": float(epoch_t1 - epoch_t0),
                    "train_images": int(len(train_items)),
                    "val_images": int(len(val_items)),
                }
            )

    return history


def save_weights(model: CNN, path: str) -> None:
    """Save model weights to a local file.

    Format: Python pickle of {name: {"shape": tuple[int,...], "data": list[float]}}.
    """
    state = {
        name: {
            "shape": tuple(t.shape()),
            "data": t.to_list(),
            "requires_grad": bool(t.has_grad()),
        }
        for name, t in model.named_tensors()
    }

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, path)


def load_weights(model: CNN, path: str) -> None:
    """Load weights previously saved by save_weights().

    If the current C++ backend exposes `tensor_write_data`, weights are written in-place.
    Otherwise, parameter tensors are recreated and assigned onto the model.
    """
    with open(path, "rb") as f:
        state = pickle.load(f)

    loaded = 0
    skipped = 0
    for name, t in model.named_tensors():
        if name not in state:
            skipped += 1
            continue
        entry = state[name]
        shape = tuple(int(x) for x in entry["shape"])
        data = entry["data"]
        req_grad = bool(entry.get("requires_grad", True))
        if tuple(t.shape()) != shape:
            raise ValueError(f"Shape mismatch for {name}: checkpoint {shape} vs model {t.shape()}")

        try:
            t.write_from_list(data)
        except RuntimeError:
            setattr(model, name, Tensor.create(data, shape, requires_grad=req_grad))
        loaded += 1

    if loaded == 0:
        raise KeyError("checkpoint did not contain any matching tensors")
    if skipped:
        print(f"load_weights: loaded={loaded}, missing_in_checkpoint={skipped} (kept defaults)")


def save_class_names(class_names: list[str], path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        import json

        json.dump(class_names, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def main() -> None:
    raise RuntimeError(
        "model.main() is disabled: dataset scanning/loading was moved to test.ipynb. "
        "In the notebook, build (train_items, val_items, class_names), preload images, "
        "then call train(model, train_items, val_items, load_arr=...)."
    )


if __name__ == "__main__":
    main()
