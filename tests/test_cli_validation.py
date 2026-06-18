import json
import os
import pytest


class TestTrainValidation:
    """train 子命令参数校验。"""

    def test_range_top_k_too_small_fails(self, cli_runner, sample_training_file):
        res = cli_runner(["train", "-i", sample_training_file, "-k", "0"])
        assert res["exit_code"] != 0
        assert "--top-k" in res["stderr"] and "不能小于" in res["stderr"]

    def test_range_top_k_too_large_fails(self, cli_runner, sample_training_file):
        res = cli_runner(["train", "-i", sample_training_file, "-k", "1000"])
        assert res["exit_code"] != 0
        assert "--top-k" in res["stderr"] and "不能大于" in res["stderr"]

    def test_range_threshold_out_of_range_high(self, cli_runner, sample_training_file):
        res = cli_runner(["train", "-i", sample_training_file, "-t", "1.5"])
        assert res["exit_code"] != 0
        assert "--threshold" in res["stderr"]

    def test_range_threshold_out_of_range_low(self, cli_runner, sample_training_file):
        res = cli_runner(["train", "-i", sample_training_file, "-t", "-0.1"])
        assert res["exit_code"] != 0
        assert "--threshold" in res["stderr"]

    def test_range_max_features_negative(self, cli_runner, sample_training_file):
        res = cli_runner(["train", "-i", sample_training_file, "--max-features", "-5"])
        assert res["exit_code"] != 0
        assert "--max-features" in res["stderr"]

    def test_missing_input_file(self, cli_runner):
        res = cli_runner(["train", "-i", "/no/such/file.json"])
        assert res["exit_code"] != 0
        assert "--input" in res["stderr"]

    def test_summary_field_without_tags_field_fails(self, cli_runner, sample_training_file):
        """互斥/依赖校验：只指定 --summary-field 不指定 --tags-field 失败。"""
        res = cli_runner([
            "train", "-i", sample_training_file,
            "--summary-field", "summary",
        ])
        assert res["exit_code"] != 0
        assert "--summary-field" in res["stderr"] and "--tags-field" in res["stderr"]

    def test_tags_field_without_summary_field_fails(self, cli_runner, sample_training_file):
        res = cli_runner([
            "train", "-i", sample_training_file,
            "--tags-field", "tags",
        ])
        assert res["exit_code"] != 0
        assert "--summary-field" in res["stderr"] and "--tags-field" in res["stderr"]

    def test_both_fields_specified_ok(self, cli_runner, sample_training_file, tmp_path):
        res = cli_runner([
            "train", "-i", sample_training_file,
            "--summary-field", "summary",
            "--tags-field", "tags",
            "-m", str(tmp_path / "m.pkl"),
        ])
        assert res["exit_code"] == 0

    def test_neither_field_specified_ok(self, cli_runner, sample_training_file, tmp_path):
        res = cli_runner([
            "train", "-i", sample_training_file,
            "-m", str(tmp_path / "m.pkl"),
        ])
        assert res["exit_code"] == 0


class TestSuggestValidation:
    """suggest 子命令参数互斥与依赖校验。"""

    def test_missing_both_document_and_input_file(self, cli_runner, tmp_path):
        model_path = tmp_path / "m.pkl"
        res = cli_runner(["suggest", "-m", str(model_path)])
        assert res["exit_code"] != 0
        assert "--document" in res["stderr"] and "--input-file" in res["stderr"]

    def test_document_and_input_file_mutually_exclusive(
        self, cli_runner, sample_training_file, sample_tag_desc_file, tmp_path
    ):
        """互斥校验：同时指定 --document 和 --input-file 必须失败。"""
        doc_file = tmp_path / "doc.txt"
        doc_file.write_text("深度学习文档", encoding="utf-8")
        # 先训练一个模型，避免因为模型不存在触发其他校验
        model_path = tmp_path / "m.pkl"
        cli_runner(["train", "-i", sample_training_file, "-m", str(model_path)])

        res = cli_runner([
            "suggest", "-m", str(model_path),
            "-d", "这是一个摘要",
            "-f", str(doc_file),
        ])
        assert res["exit_code"] != 0
        assert "--document" in res["stderr"] and "--input-file" in res["stderr"]
        assert "互斥" in res["stderr"]

    def test_explain_words_requires_explain_flag(
        self, cli_runner, sample_training_file, tmp_path
    ):
        """依赖校验：--explain-words 必须伴随 --explain。"""
        model_path = tmp_path / "m.pkl"
        cli_runner(["train", "-i", sample_training_file, "-m", str(model_path)])
        res = cli_runner([
            "suggest", "-m", str(model_path),
            "-d", "文档摘要",
            "--explain-words", "5",
        ])
        assert res["exit_code"] != 0
        assert "--explain-words" in res["stderr"] and "--explain" in res["stderr"]

    def test_explain_with_words_ok(self, cli_runner, sample_training_file, tmp_path):
        model_path = tmp_path / "m.pkl"
        cli_runner(["train", "-i", sample_training_file, "-m", str(model_path)])
        res = cli_runner([
            "suggest", "-m", str(model_path),
            "-d", "深度学习文档",
            "--explain", "--explain-words", "2",
        ])
        assert res["exit_code"] == 0

    def test_cold_start_requires_tag_descriptions(
        self, cli_runner, tmp_path
    ):
        """模型文件不存在时必须提供 --tag-descriptions 才能冷启动。"""
        non_existent = tmp_path / "nope.pkl"
        res = cli_runner([
            "suggest", "-m", str(non_existent),
            "-d", "一些摘要",
        ])
        assert res["exit_code"] != 0
        assert "冷启动" in res["stderr"] and "--tag-descriptions" in res["stderr"]

    def test_cold_start_with_tag_descriptions_ok(
        self, cli_runner, sample_tag_desc_file, tmp_path
    ):
        non_existent = tmp_path / "nope.pkl"
        res = cli_runner([
            "suggest", "-m", str(non_existent),
            "-d", "深度学习相关内容",
            "--tag-descriptions", sample_tag_desc_file,
            "-k", "2", "-t", "0.0",
        ])
        assert res["exit_code"] == 0
        assert "推荐标签" in res["stdout"]

    def test_suggest_top_k_range(self, cli_runner, sample_training_file, tmp_path):
        model_path = tmp_path / "m.pkl"
        cli_runner(["train", "-i", sample_training_file, "-m", str(model_path)])
        res = cli_runner([
            "suggest", "-m", str(model_path),
            "-d", "文档", "-k", "-3",
        ])
        assert res["exit_code"] != 0
        assert "--top-k" in res["stderr"]

    def test_suggest_threshold_range(self, cli_runner, sample_training_file, tmp_path):
        model_path = tmp_path / "m.pkl"
        cli_runner(["train", "-i", sample_training_file, "-m", str(model_path)])
        res = cli_runner([
            "suggest", "-m", str(model_path),
            "-d", "文档", "-t", "2.0",
        ])
        assert res["exit_code"] != 0
        assert "--threshold" in res["stderr"]


class TestFeedbackListValidation:
    """feedback list 子命令参数校验。"""

    def test_limit_zero_fails(self, cli_runner):
        res = cli_runner(["feedback", "--backend", "json", "-s", "/tmp/nonexist.json",
                          "list", "-n", "0"])
        assert res["exit_code"] != 0
        assert "--limit" in res["stderr"]

    def test_offset_negative_fails(self, cli_runner):
        res = cli_runner(["feedback", "--backend", "json", "-s", "/tmp/nonexist.json",
                          "list", "--offset", "-5"])
        assert res["exit_code"] != 0
        assert "--offset" in res["stderr"]

    def test_valid_offset_and_limit_ok(self, tmp_path):
        """确认合法的 limit/offset 不被拦截。"""
        from feedback_store import create_feedback_store
        fp = tmp_path / "fb.json"
        store = create_feedback_store("json", filepath=str(fp))
        store.add_feedback("d1", ["a"], [], [], [])
        records = store.list_feedback(limit=5, offset=0)
        assert len(records) == 1


class TestMigrateValidation:
    """migrate 子命令参数校验。"""

    def test_status_on_nonexistent_db_creates_empty_schema(self, tmp_path, cli_runner):
        db_path = tmp_path / "new.db"
        assert not os.path.exists(db_path)
        res = cli_runner(["migrate", "--db-path", str(db_path), "status"])
        assert res["exit_code"] == 0
        assert os.path.exists(db_path)
        assert "当前 schema 版本" in res["stdout"]

    def test_upgrade_on_nonexistent_db_fails(self, tmp_path, cli_runner):
        db_path = tmp_path / "new.db"
        res = cli_runner(["migrate", "--db-path", str(db_path), "upgrade"])
        assert res["exit_code"] != 0
        assert "--db-path" in res["stderr"] and "不存在" in res["stderr"]

    def test_backfill_on_nonexistent_db_fails(self, tmp_path, cli_runner):
        db_path = tmp_path / "new.db"
        res = cli_runner(["migrate", "--db-path", str(db_path), "backfill"])
        assert res["exit_code"] != 0

    def test_rollback_requires_target_version(self, populated_db_path, cli_runner):
        db_path, _, _ = populated_db_path
        # 不指定 --target-version 通过 argparse 直接拦截 -> exit 2
        res = cli_runner(["migrate", "--db-path", db_path, "rollback"])
        assert res["exit_code"] != 0

    def test_rollback_negative_version_fails(self, populated_db_path, cli_runner):
        db_path, _, _ = populated_db_path
        res = cli_runner(["migrate", "--db-path", db_path, "rollback",
                          "--target-version", "-1"])
        assert res["exit_code"] != 0
        assert "--target-version" in res["stderr"]

    def test_upgrade_target_less_than_current_fails(self, populated_db_path, cli_runner):
        db_path, _, _ = populated_db_path
        res = cli_runner(["migrate", "--db-path", db_path, "upgrade",
                          "--target-version", "0"])
        assert res["exit_code"] != 0

    def test_status_roundtrip(self, populated_db_path, cli_runner):
        db_path, _, _ = populated_db_path
        res = cli_runner(["migrate", "--db-path", db_path, "status"])
        assert res["exit_code"] == 0
        assert "当前 schema 版本: 2" in res["stdout"]

    def test_cli_rollback_and_upgrade_roundtrip(self, populated_db_path, cli_runner):
        """通过 CLI 回滚到 v1，再升级到 v2，数据不应丢失。"""
        db_path, fb1, fb2 = populated_db_path
        res_rb = cli_runner(["migrate", "--db-path", db_path, "rollback",
                             "--target-version", "1"])
        assert res_rb["exit_code"] == 0
        assert "当前 schema 版本: 1" in res_rb["stdout"]

        res_up = cli_runner(["migrate", "--db-path", db_path, "upgrade"])
        assert res_up["exit_code"] == 0
        assert "当前 schema 版本: 2" in res_up["stdout"]

        from feedback_store import SqliteFeedbackStore
        store = SqliteFeedbackStore(db_path=db_path)
        res = cli_runner(["migrate", "--db-path", db_path, "backfill"])
        assert res["exit_code"] == 0
        accepted = store.get_tags_by_type("accepted")
        assert len(accepted) >= 2


class TestRetrainValidation:
    """retrain 子命令参数范围。"""

    def test_retrain_threshold_oor(self, cli_runner):
        res = cli_runner(["retrain", "-t", "3.0"])
        assert res["exit_code"] != 0
        assert "--threshold" in res["stderr"]

    def test_retrain_top_k_oor(self, cli_runner):
        res = cli_runner(["retrain", "-k", "0"])
        assert res["exit_code"] != 0
        assert "--top-k" in res["stderr"]


class TestIntegratedEndToEnd:
    """端到端：CLI 命令链互相配合。"""

    def test_train_suggest_feedback_retrain_pipeline(
        self, cli_runner, sample_training_file, tmp_path
    ):
        model_path = tmp_path / "m.pkl"
        fb_json = tmp_path / "fb.json"

        # 1. train
        r1 = cli_runner(["train", "-i", sample_training_file, "-m", str(model_path),
                         "-k", "3", "-t", "0.05"])
        assert r1["exit_code"] == 0

        # 2. suggest
        r2 = cli_runner(["suggest", "-m", str(model_path),
                         "-d", "深度学习图像识别应用", "-k", "3"])
        assert r2["exit_code"] == 0
        assert "推荐标签" in r2["stdout"]

        # 3. feedback add (json backend)
        r3 = cli_runner([
            "feedback", "--backend", "json", "-s", str(fb_json),
            "add", "-d", "深度学习图像识别应用",
            "--suggested", "深度学习,图像识别,数据挖掘",
            "-a", "深度学习,图像识别",
            "-r", "数据挖掘",
            "--added", "CV",
        ])
        assert r3["exit_code"] == 0

        # 4. feedback stats
        r4 = cli_runner([
            "feedback", "--backend", "json", "-s", str(fb_json), "stats",
        ])
        assert r4["exit_code"] == 0
        assert "total" in r4["stdout"]

        # 5. retrain
        r5 = cli_runner([
            "retrain", "-m", str(model_path),
            "--backend", "json", "-s", str(fb_json),
        ])
        assert r5["exit_code"] == 0
