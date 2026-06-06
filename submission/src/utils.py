"""Seed control, logging, timers, config loading."""
from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml


def set_seeds(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def get_logger(name: str = "submission") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def timer(label: str, sink: list | None = None):
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    msg = f"[{label}] {elapsed:.2f}s"
    get_logger().info(msg)
    if sink is not None:
        sink.append({"label": label, "seconds": elapsed})
