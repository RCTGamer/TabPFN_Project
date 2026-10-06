import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


@pytest.fixture(scope="session")
def device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def breast_cancer_split():
    """Small real dataset (569 x 30, binary) - fast enough for CPU runs."""
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


@pytest.fixture(scope="session")
def redundant_split(breast_cancer_split):
    """Breast cancer data padded with constant, duplicated-column and duplicated-row noise.

    The extra columns/rows carry no new information, so a good preprocessing
    step should remove them without changing accuracy.
    """
    X_train, X_test, y_train, y_test = breast_cancer_split

    def pad(X):
        constant = np.ones((len(X), 5))
        copies = X[:, :10] * 1.0001  # ~perfectly correlated with originals
        return np.hstack([X, constant, copies])

    X_train, X_test = pad(X_train), pad(X_test)
    X_train = np.vstack([X_train, X_train[:100]])
    y_train = np.concatenate([y_train, y_train[:100]])
    return X_train, X_test, y_train, y_test


@pytest.fixture(scope="session")
def tabpfn_available(device, breast_cancer_split):
    """Skip model tests when TabPFN weights can't be loaded.

    The default weights are gated on Hugging Face: accept the license at
    https://huggingface.co/Prior-Labs and set HF_TOKEN (or run `hf auth login`).
    """
    from tabpfn import TabPFNClassifier
    from tabpfn.errors import TabPFNError

    X_train, _, y_train, _ = breast_cancer_split
    try:
        TabPFNClassifier(device=device, n_estimators=1).fit(X_train[:50], y_train[:50])
    except (TabPFNError, OSError) as exc:
        pytest.skip(f"TabPFN weights unavailable: {type(exc).__name__}")
    return True
