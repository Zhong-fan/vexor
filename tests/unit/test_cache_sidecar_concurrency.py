"""Exercise sidecar ownership with separate processes sharing one SQLite cache."""

from __future__ import annotations

import multiprocessing
import sqlite3
from pathlib import Path

import pytest

from vexor import cache
from vexor.services.cache_service import load_index_metadata_safe


def _options(root: Path) -> dict:
    return {
        "root": root,
        "model": "model",
        "include_hidden": False,
        "mode": "name",
        "recursive": True,
    }


def _entry(root: Path, value: float) -> cache.IndexedChunk:
    return cache.IndexedChunk(root / "a.txt", "a.txt", 0, "text", [value, 1.0])


def _write_paused(cache_dir, root, incremental, stage, paused, release):
    with cache.cache_dir_context(cache_dir):
        if stage == "temporary":
            original = cache.np.lib.format.open_memmap

            def pause(*args, **kwargs):
                result = original(*args, **kwargs)
                if kwargs.get("mode") == "w+":
                    paused.set()
                    if not release.wait(10):
                        raise TimeoutError("Writer was not released")
                return result

            cache.np.lib.format.open_memmap = pause
        else:
            original = cache._write_vector_file

            def pause(*args, **kwargs):
                result = original(*args, **kwargs)
                paused.set()
                if not release.wait(10):
                    raise TimeoutError("Writer was not released")
                return result

            cache._write_vector_file = pause
        if incremental:
            cache.apply_index_updates(
                **_options(root),
                ordered_entries=[("a.txt", 0)],
                changed_entries=[_entry(root, 2.0)],
                removed_rel_paths=[],
            )
        else:
            cache.store_index(**_options(root), entries=[_entry(root, 2.0)])


def _read_and_prune(cache_dir, root, started, finished):
    with cache.cache_dir_context(cache_dir):
        started.set()
        cache.load_index_vectors(
            root, "model", False, "name", True, memory_cache=cache.IndexVectorCache()
        )
        finished.set()


@pytest.mark.parametrize("incremental", [False, True])
@pytest.mark.parametrize("stage", ["temporary", "final"])
def test_pruning_preserves_other_process_pending_generation(tmp_path, incremental, stage):
    cache_dir = tmp_path / "shared-cache"
    roots = [tmp_path / "a", tmp_path / "b"]
    for root in roots:
        root.mkdir()
        (root / "a.txt").write_text("text", encoding="utf-8")
    with cache.cache_dir_context(cache_dir):
        for root in roots:
            cache.store_index(**_options(root), entries=[_entry(root, 1.0)])
    orphan = cache_dir / "vectors" / "orphan.npy"
    orphan.write_bytes(b"abandoned")
    crashed_temporary = cache_dir / "vectors" / ".crashed.tmp.npy"
    crashed_temporary.write_bytes(b"abandoned temporary file")
    ctx = multiprocessing.get_context("spawn")
    paused, release, started, finished = [ctx.Event() for _ in range(4)]
    writer = ctx.Process(
        target=_write_paused, args=(cache_dir, roots[0], incremental, stage, paused, release)
    )
    pruner = ctx.Process(target=_read_and_prune, args=(cache_dir, roots[1], started, finished))
    writer.start()
    try:
        assert paused.wait(10), "Writer did not reach pending sidecar"
        pruner.start()
        assert started.wait(10), "Pruner did not start"
        assert finished.wait(10), "Healthy reads must succeed while cleanup is deferred"
        pending = (
            list((cache_dir / "vectors").glob(".*.tmp.npy"))
            if stage == "temporary"
            else list((cache_dir / "vectors").glob("*.npy"))
        )
        assert len(pending) == (2 if stage == "temporary" else 5)
        assert orphan.exists(), "Cleanup must defer while a pending sidecar is uncommitted"
        assert crashed_temporary.exists()
    finally:
        release.set()
        writer.join(10)
        if pruner.pid is not None:
            pruner.join(10)
        for process in (writer, pruner):
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(5)
    assert writer.exitcode == pruner.exitcode == 0
    with cache.cache_dir_context(cache_dir):
        for root, expected in zip(roots, [2.0, 1.0], strict=True):
            _, vectors, _ = cache.load_index_vectors(root, "model", False, "name", True)
            assert vectors.tolist() == [[expected, 1.0]]
            del vectors
    assert not orphan.exists()
    assert not crashed_temporary.exists()
    assert not list((cache_dir / "vectors").glob(".*.tmp.npy"))


def test_healthy_memory_cache_read_defers_cleanup_under_write_lock(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("text", encoding="utf-8")
    with cache.cache_dir_context(tmp_path / "cache"):
        cache.store_index(**_options(root), entries=[_entry(root, 1.0)])
        orphan = tmp_path / "cache" / "vectors" / "orphan.npy"
        orphan.write_bytes(b"abandoned")
        blocker = cache._connect(cache.cache_db_path())
        try:
            blocker.execute("BEGIN IMMEDIATE;")
            memory_cache = cache.IndexVectorCache()
            _, vectors, _ = cache.load_index_vectors(
                root, "model", False, "name", True, memory_cache=memory_cache
            )
            assert vectors.tolist() == [[1.0, 1.0]]
            assert orphan.exists()
        finally:
            blocker.rollback()
            blocker.close()
        memory_cache.prune()
        assert not orphan.exists()


def test_readonly_database_can_load_into_memory_cache(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("text", encoding="utf-8")
    with cache.cache_dir_context(tmp_path / "cache"):
        cache.store_index(**_options(root), entries=[_entry(root, 1.0)])
        connect = cache._connect
        monkeypatch.setattr(cache, "_connect", lambda path, **kwargs: connect(path, readonly=True))
        _, vectors, _ = cache.load_index_vectors(
            root, "model", False, "name", True, memory_cache=cache.IndexVectorCache()
        )
        assert vectors.tolist() == [[1.0, 1.0]]


@pytest.mark.parametrize("incremental", [False, True])
def test_writer_lock_timeout_preserves_existing_index(
    tmp_path, monkeypatch, incremental,
):
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("text", encoding="utf-8")
    with cache.cache_dir_context(tmp_path / "cache"):
        cache.store_index(**_options(root), entries=[_entry(root, 1.0)])
        connect = cache._connect

        def no_wait(path, **kwargs):
            conn = connect(path, **kwargs)
            conn.execute("PRAGMA busy_timeout = 0;")
            return conn

        monkeypatch.setattr(cache, "_connect", no_wait)
        blocker = connect(cache.cache_db_path())
        blocker.execute("BEGIN IMMEDIATE;")
        try:
            with pytest.raises(sqlite3.OperationalError, match="database is locked"):
                if incremental:
                    cache.apply_index_updates(
                        **_options(root), ordered_entries=[("a.txt", 0)],
                        changed_entries=[_entry(root, 2.0)], removed_rel_paths=[],
                    )
                else:
                    cache.store_index(**_options(root), entries=[_entry(root, 2.0)])
        finally:
            blocker.rollback()
            blocker.close()
        _, vectors, _ = cache.load_index_vectors(root, "model", False, "name", True)
        assert vectors.tolist() == [[1.0, 1.0]]


@pytest.mark.parametrize("command", ["index", "search", "json", "clear"])
def test_cli_reports_cache_write_timeout_without_traceback(tmp_path, monkeypatch, command):
    from typer.testing import CliRunner

    from vexor import cli, config

    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("text", encoding="utf-8")
    monkeypatch.setenv("VEXOR_CONFIG_JSON", '{"provider":"local","model":"model"}')
    with (
        cache.cache_dir_context(tmp_path / "cache"),
        config.config_dir_context(tmp_path / "config"),
    ):
        cache.store_index(**_options(root), entries=[_entry(root, 1.0)])
        connect = cache._connect

        def no_wait(path, **kwargs):
            conn = connect(path, **kwargs)
            conn.execute("PRAGMA busy_timeout = 0;")
            return conn

        def write_index(*args, **kwargs):
            cache.store_index(**_options(root), entries=[_entry(root, 2.0)])

        monkeypatch.setattr(cache, "_connect", no_wait)
        monkeypatch.setattr(cli, "build_index", write_index)
        monkeypatch.setattr(cli, "perform_search", write_index)
        blocker = connect(cache.cache_db_path())
        blocker.execute("BEGIN IMMEDIATE;")
        try:
            args = ["index"] if command in {"index", "clear"} else ["search", "query"]
            args.extend(["--path", str(root), "--mode", "name"])
            if command == "json":
                args.append("--format=json")
            if command == "clear":
                args.append("--clear")
            result = CliRunner().invoke(cli.app, args)
        finally:
            blocker.rollback()
            blocker.close()
        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert "retry" in result.output
        assert "Traceback" not in result.output
        if command == "json":
            assert result.stdout == ""
            assert "retry" in result.stderr


def test_cli_timeout_from_embedding_cache_has_retry_hint(tmp_path, monkeypatch):
    import importlib

    import numpy as np
    from typer.testing import CliRunner

    from vexor import cli, config

    class OfflineSearcher:
        def __init__(self, **kwargs):
            pass

        def embed_texts(self, labels):
            return np.ones((len(labels), 2), dtype=np.float32)

    root, peer = tmp_path / "new-project", tmp_path / "peer"
    for project in (root, peer):
        project.mkdir()
        (project / "a.txt").write_text("text", encoding="utf-8")
    monkeypatch.setenv(
        "VEXOR_CONFIG_JSON", '{"provider":"local","model":"model","update_check":false}',
    )
    monkeypatch.setattr(importlib.import_module("vexor.search"), "VexorSearcher", OfflineSearcher)
    with (
        cache.cache_dir_context(tmp_path / "cache"),
        config.config_dir_context(tmp_path / "config"),
    ):
        cache.store_index(**_options(peer), entries=[_entry(peer, 1.0)])
        connect = cache._connect

        def no_wait(path, **kwargs):
            conn = connect(path, **kwargs)
            conn.execute("PRAGMA busy_timeout = 0;")
            return conn

        monkeypatch.setattr(cache, "_connect", no_wait)
        blocker = connect(cache.cache_db_path())
        blocker.execute("BEGIN IMMEDIATE;")
        try:
            result = CliRunner().invoke(cli.app, ["index", "--path", str(root), "--mode", "name"])
        finally:
            blocker.rollback()
            blocker.close()
        assert result.exit_code == 1
        assert isinstance(result.exception, SystemExit)
        assert "retry" in result.stdout
        _, vectors, _ = cache.load_index_vectors(peer, "model", False, "name", True)
        assert vectors.tolist() == [[1.0, 1.0]]


@pytest.mark.parametrize("command", ["index", "search", "json"])
def test_cli_does_not_relabel_other_sqlite_errors_as_busy(tmp_path, monkeypatch, command):
    from contextlib import closing

    from typer.testing import CliRunner

    from vexor import cli

    monkeypatch.setenv("VEXOR_CONFIG_JSON", '{"provider":"local","model":"model"}')

    def invalid_query(*args, **kwargs):
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.execute("SELECT nonexistent_column")

    monkeypatch.setattr(cli, "build_index", invalid_query)
    monkeypatch.setattr(cli, "perform_search", invalid_query)
    args = ["index"] if command == "index" else ["search", "query"]
    args.extend(["--path", str(tmp_path), "--mode", "name"])
    if command == "json":
        args.append("--format=json")
    result = CliRunner().invoke(cli.app, args)
    assert isinstance(result.exception, sqlite3.OperationalError)
    assert "no such column" in str(result.exception)
    assert "retry" not in result.output


@pytest.mark.parametrize("empty", [False, True])
def test_committed_missing_sidecar_is_damage_not_missing_index(tmp_path, empty):
    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("text", encoding="utf-8")
    with cache.cache_dir_context(tmp_path / "cache"):
        cache.store_index(**_options(root), entries=[] if empty else [_entry(root, 1.0)])
        sidecar = next((tmp_path / "cache" / "vectors").glob("*.npy"))
        sidecar.unlink()
        for load in (
            lambda: cache.load_index_vectors(root, "model", False, "name", True),
            lambda: load_index_metadata_safe(root, "model", False, True, "name", True),
        ):
            with pytest.raises(RuntimeError, match="vexor index --clear") as error:
                load()
            assert str(sidecar) in str(error.value)


@pytest.mark.parametrize("command", ["index", "show", "search", "json"])
def test_cli_reports_committed_sidecar_damage_and_can_clear(tmp_path, monkeypatch, command):
    from typer.testing import CliRunner

    from vexor import cli, config

    root = tmp_path / "project"
    root.mkdir()
    (root / "a.txt").write_text("text", encoding="utf-8")
    monkeypatch.setenv("VEXOR_CONFIG_JSON", '{"provider":"local","model":"model"}')
    with (
        cache.cache_dir_context(tmp_path / "cache"),
        config.config_dir_context(tmp_path / "config"),
    ):
        cache.store_index(**_options(root), entries=[_entry(root, 1.0)])
        next((tmp_path / "cache" / "vectors").glob("*.npy")).unlink()
        args = ["search", "query"] if command in {"search", "json"} else ["index"]
        args.extend(["--path", str(root), "--mode", "name"])
        if command == "show":
            args.append("--show")
        if command == "json":
            args.append("--format=json")
        result = CliRunner().invoke(cli.app, args)
        assert result.exit_code == 1, result.output
        assert "Damaged index cache" in result.output
        assert "vexor index --clear" in " ".join(result.output.split())
        if command == "json":
            assert result.stdout == ""
            assert "Damaged index cache" in result.stderr
        result = CliRunner().invoke(
            cli.app, ["index", "--path", str(root), "--mode", "name", "--clear"]
        )
        assert result.exit_code == 0, result.output
        assert load_index_metadata_safe(root, "model", False, True, "name", True) is None
