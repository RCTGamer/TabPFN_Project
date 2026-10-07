"""Keyed npz store for per-estimator outputs. Writes are atomic (temp file + rename)."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

_META = "__meta__"


def _canonical(obj):
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class OutputCache:
    def __init__(self, root):
        self.root = Path(root)

    @staticmethod
    def make_key(
        dataset,
        task,
        split_seed,
        n_train,
        n_test,
        backend: dict,
        n_estimators,
        subset_id="full",
        seed=None,
        **extra,
    ) -> str:
        """Hash of everything that determines the stored outputs. Dict ordering does not matter."""
        payload = _canonical(
            {
                "dataset": dataset,
                "task": task,
                "split_seed": split_seed,
                "seed": split_seed if seed is None else seed,
                "n_train": n_train,
                "n_test": n_test,
                "backend": backend,
                "n_estimators": n_estimators,
                "subset_id": subset_id,
                **extra,
            }
        )
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.npz"

    def exists(self, key: str) -> bool:
        return self.path(key).is_file()

    def put(self, key: str, arrays: dict, meta: dict | None = None) -> None:
        final = self.path(key)
        final.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: np.asarray(v) for k, v in arrays.items()}
        payload[_META] = np.array(json.dumps(_canonical(meta or {})))
        fd, tmp = tempfile.mkstemp(dir=final.parent, prefix=f".{key}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                np.savez(f, **payload)
            os.replace(tmp, final)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    def get(self, key: str) -> tuple[dict, dict]:
        """(arrays, meta)."""
        with np.load(self.path(key), allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files if k != _META}
            meta = json.loads(str(z[_META])) if _META in z.files else {}
        return arrays, meta
