import os
import json
import sqlite3
import pytest

from db_session import SqliteSession
from config_loader import AppConfig, _BUILTIN_DEFAULTS


class TestSqliteSession:
    """db_session.SqliteSession 功能测试。"""

    def test_connect_auto_creates_dir(self, tmp_path):
        """父目录不存在时应自动创建。"""
        db_path = str(tmp_path / "sub" / "nested" / "x.db")
        session = SqliteSession(db_path=db_path)
        conn = session.connect()
        conn.close()
        assert os.path.exists(db_path)

    def test_pragmas_applied(self, tmp_db_path):
        session = SqliteSession(db_path=tmp_db_path)
        with session.readonly() as cursor:
            cursor.execute("PRAGMA journal_mode")
            mode = cursor.fetchone()[0]
        assert mode.upper() == "WAL"

    def test_transaction_commit_on_success(self, tmp_db_path):
        session = SqliteSession(db_path=tmp_db_path)
        with session.transaction() as cursor:
            cursor.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
            cursor.execute("INSERT INTO t VALUES (1, 'a')")
        with session.readonly() as cursor:
            cursor.execute("SELECT v FROM t WHERE id = 1")
            assert cursor.fetchone()[0] == "a"

    def test_transaction_rollback_on_error(self, tmp_db_path):
        session = SqliteSession(db_path=tmp_db_path)
        with session.transaction() as cursor:
            cursor.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        try:
            with session.transaction() as cursor:
                cursor.execute("INSERT INTO t VALUES (1)")
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        with session.readonly() as cursor:
            cursor.execute("SELECT COUNT(*) FROM t")
            assert cursor.fetchone()[0] == 0

    def test_fetchone_and_fetchall(self, tmp_db_path):
        session = SqliteSession(db_path=tmp_db_path)
        with session.transaction() as cursor:
            cursor.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
            cursor.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
        assert session.fetchone("SELECT MIN(id) FROM t")[0] == 1
        assert len(session.fetchall("SELECT id FROM t")) == 3

    def test_table_exists(self, tmp_db_path):
        session = SqliteSession(db_path=tmp_db_path)
        assert not session.table_exists("missing")
        with session.transaction() as cursor:
            cursor.execute("CREATE TABLE real (id INTEGER)")
        assert session.table_exists("real")

    def test_column_exists(self, tmp_db_path):
        session = SqliteSession(db_path=tmp_db_path)
        with session.transaction() as cursor:
            cursor.execute("CREATE TABLE t (a INTEGER, b TEXT)")
        assert session.column_exists("t", "a")
        assert not session.column_exists("t", "c")
        assert not session.column_exists("missing", "a")

    def test_executemany(self, tmp_db_path):
        session = SqliteSession(db_path=tmp_db_path)
        with session.transaction() as cursor:
            cursor.execute("CREATE TABLE t (id INTEGER PRIMARY KEY)")
        session.executemany("INSERT INTO t VALUES (?)", [(10,), (20,)])
        rows = session.fetchall("SELECT id FROM t ORDER BY id")
        assert [r[0] for r in rows] == [10, 20]


class TestAppConfig:
    """config_loader.AppConfig 功能测试。"""

    def test_load_default_builtin_when_no_file(self, tmp_path, monkeypatch):
        """找不到外部配置时使用内置默认值。"""
        monkeypatch.chdir(tmp_path)
        cfg = AppConfig.load()
        assert cfg.get("model.default_top_k") == _BUILTIN_DEFAULTS["model"]["default_top_k"]
        assert cfg.get("validation.threshold.min") == 0.0

    def test_load_explicit_path(self, tmp_path):
        data = {"model": {"default_top_k": 7}, "custom": {"nested": {"v": 99}}}
        f = tmp_path / "c.json"
        f.write_text(json.dumps(data), encoding="utf-8")
        cfg = AppConfig.load(str(f))
        assert cfg.get("model.default_top_k") == 7
        assert cfg.get("custom.nested.v") == 99
        assert cfg.get("no.such.key", "fallback") == "fallback"

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AppConfig.load(str(tmp_path / "nope.json"))

    def test_get_range(self, tmp_path):
        cfg = AppConfig.load()
        min_val, max_val = cfg.get_range("validation.top_k")
        assert min_val == 1 and max_val == 100

    def test_get_range_unknown_key_returns_none(self, tmp_path):
        cfg = AppConfig.load()
        assert cfg.get_range("missing.range") == (None, None)

    def test_as_dict_returns_deep_copy(self, tmp_path):
        cfg = AppConfig.load()
        d1 = cfg.as_dict()
        d2 = cfg.as_dict()
        d1["model"]["default_top_k"] = 999
        assert d2["model"]["default_top_k"] != 999


class TestConfigFromCli:
    """CLI --config 选项能正确覆盖默认配置。"""

    def test_cli_config_affects_defaults(self, cli_runner, sample_training_file, tmp_path):
        """自定义配置里把默认 top_k 改成 2，help 里应看到对应值。"""
        custom_cfg = {
            "model": {
                "default_model_path": "my_model.pkl",
                "default_top_k": 2,
                "default_threshold": 0.5,
                "max_features": 1000,
                "default_fields": {"summary_field": "text", "tags_field": "labels"},
            },
            "suggest": {"default_top_k": 2, "default_threshold": 0.5, "explain_words": 2},
            "validation": _BUILTIN_DEFAULTS["validation"],
            "feedback": _BUILTIN_DEFAULTS["feedback"],
            "migrate": _BUILTIN_DEFAULTS["migrate"],
            "cli": {"prog_name": "tag-suggester", "description": "custom desc"},
        }
        cfg_path = tmp_path / "custom.json"
        cfg_path.write_text(json.dumps(custom_cfg, ensure_ascii=False), encoding="utf-8")

        res = cli_runner(["-C", str(cfg_path), "train", "--help"])
        assert res["exit_code"] == 0
        assert "默认: 2" in res["stdout"]
        assert "默认: 0.5" in res["stdout"]
        assert "默认: my_model.pkl" in res["stdout"]
