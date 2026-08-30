"""Configuration path regressions."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

import config as config_module

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]


def test_default_paths_are_independent_of_current_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RAG_DATA_PATH", raising=False)
    monkeypatch.delenv("RAG_INDEX_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    reloaded = importlib.reload(config_module)
    config = reloaded.RAGConfig.from_env()

    assert Path(config.data_path) == WORKSPACE_ROOT / "data" / "C8" / "cook"
    assert Path(config.index_save_path) == PROJECT_ROOT / "vector_index"
    assert Path(config.data_path).is_absolute()
    assert Path(config.index_save_path).is_absolute()


def test_relative_environment_paths_resolve_from_project_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RAG_DATA_PATH", "fixtures/cook")
    monkeypatch.setenv("RAG_INDEX_PATH", "artifacts/index")

    config = config_module.RAGConfig.from_env()

    assert Path(config.data_path) == PROJECT_ROOT / "fixtures" / "cook"
    assert Path(config.index_save_path) == PROJECT_ROOT / "artifacts" / "index"
