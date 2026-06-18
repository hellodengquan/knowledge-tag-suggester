import os
import sqlite3
import pytest

from feedback_store import SqliteFeedbackStore, SCHEMA_MIGRATIONS, CURRENT_SCHEMA_VERSION


class TestMigrationForward:
    """测试前向迁移 (forward / upgrade) 路径。"""

    def test_fresh_db_auto_upgrade_to_latest(self, tmp_db_path):
        """新建数据库应自动升级到 CURRENT_SCHEMA_VERSION。"""
        store = SqliteFeedbackStore(db_path=tmp_db_path)
        assert store.get_schema_version() == CURRENT_SCHEMA_VERSION

    def test_schema_version_table_exists(self, tmp_db_path):
        """schema_version 表应被正确创建并有对应记录。"""
        SqliteFeedbackStore(db_path=tmp_db_path)
        conn = sqlite3.connect(tmp_db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'")
        assert cur.fetchone() is not None
        cur.execute("SELECT version FROM schema_version ORDER BY version")
        versions = [r[0] for r in cur.fetchall()]
        conn.close()
        assert versions == list(range(1, CURRENT_SCHEMA_VERSION + 1))

    def test_feedback_table_created_v1(self, tmp_db_path):
        """v1 迁移应创建 feedback 表，包含所有必要列。"""
        SqliteFeedbackStore(db_path=tmp_db_path)
        conn = sqlite3.connect(tmp_db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(feedback)")
        columns = {row[1]: row[2] for row in cur.fetchall()}
        conn.close()
        for col in ["id", "document", "suggested_tags", "accepted_tags",
                    "rejected_tags", "user_added_tags", "final_tags",
                    "timestamp", "metadata"]:
            assert col in columns

    def test_feedback_tag_table_created_v2(self, tmp_db_path):
        """v2 迁移应创建 feedback_tag 表和两个索引。"""
        SqliteFeedbackStore(db_path=tmp_db_path)
        conn = sqlite3.connect(tmp_db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(feedback_tag)")
        columns = {row[1]: row[2] for row in cur.fetchall()}
        for col in ["feedback_id", "tag", "tag_type"]:
            assert col in columns
        cur.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [r[0] for r in cur.fetchall()]
        conn.close()
        assert "idx_feedback_tag_type" in indexes
        assert "idx_feedback_tag_name" in indexes

    def test_add_feedback_writes_both_tables(self, tmp_db_path):
        """v2 下添加反馈时 feedback_tag 表应同步写入。"""
        store = SqliteFeedbackStore(db_path=tmp_db_path)
        store.add_feedback(
            document="测试文档",
            suggested_tags=["A", "B"],
            accepted_tags=["A"],
            rejected_tags=["B"],
            user_added_tags=["C"],
        )
        conn = sqlite3.connect(tmp_db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM feedback")
        assert cur.fetchone()[0] == 1
        cur.execute("SELECT tag, tag_type FROM feedback_tag ORDER BY tag_type, tag")
        rows = cur.fetchall()
        conn.close()
        expected = {
            ("A", "accepted"), ("B", "rejected"), ("C", "added"),
            ("A", "B", "C", "suggested"), ("A", "C", "final"),
        }
        simplified = {(tag, tag_type) for tag, tag_type in rows}
        for t in ["accepted", "rejected", "added", "suggested", "final"]:
            assert any(tag_type == t for _, tag_type in rows)

    def test_get_tags_by_type_after_add(self, populated_db_path):
        """get_tags_by_type 应基于 feedback_tag 索引表正确返回。"""
        db_path, _, _ = populated_db_path
        store = SqliteFeedbackStore(db_path=db_path)
        accepted = store.get_tags_by_type("accepted")
        assert len(accepted) >= 2
        assert any(r["tag"] == "深度学习" for r in accepted)
        assert any(r["tag"] == "React" for r in accepted)

    def test_upgrade_already_latest_is_noop(self, tmp_db_path):
        """已是最新版本时 upgrade() 返回 0。"""
        store = SqliteFeedbackStore(db_path=tmp_db_path)
        applied = store.upgrade()
        assert applied == 0

    def test_get_schema_version_for_empty_db_is_zero(self, tmp_db_path):
        """纯空数据库 (未 init) 时通过内部方式版本为 0。"""
        SqliteFeedbackStore(db_path=tmp_db_path)
        conn = sqlite3.connect(tmp_db_path)
        cur = conn.cursor()
        cur.execute("DELETE FROM schema_version")
        conn.commit()
        conn.close()
        store2 = SqliteFeedbackStore(db_path=tmp_db_path)
        assert store2.get_schema_version() == CURRENT_SCHEMA_VERSION


class TestMigrationRollback:
    """测试回滚 (rollback) 路径。"""

    def test_rollback_to_v1_removes_v2_objects(self, tmp_db_path):
        """回滚到 v1 应删除 feedback_tag 表和索引，但保留 feedback 表。"""
        store = SqliteFeedbackStore(db_path=tmp_db_path)
        assert store.get_schema_version() == 2
        reverted = store.rollback(target_version=1)
        assert reverted == 1
        assert store.get_schema_version() == 1

        conn = sqlite3.connect(tmp_db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback_tag'")
        assert cur.fetchone() is None
        cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_feedback_tag_%'")
        assert cur.fetchone() is None
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'")
        assert cur.fetchone() is not None
        cur.execute("SELECT version FROM schema_version")
        versions = [r[0] for r in cur.fetchall()]
        conn.close()
        assert 1 in versions
        assert 2 not in versions

    def test_rollback_to_0_removes_everything(self, tmp_db_path):
        """完全回滚 (target=0) 应清理所有对象。"""
        store = SqliteFeedbackStore(db_path=tmp_db_path)
        store.add_feedback("doc", ["a"], [], ["a"], [])
        reverted = store.rollback(target_version=0)
        assert reverted == 2
        assert store.get_schema_version() == 0

    def test_rollback_to_same_version_is_noop(self, tmp_db_path):
        """回滚到当前版本应返回 0。"""
        store = SqliteFeedbackStore(db_path=tmp_db_path)
        reverted = store.rollback(target_version=CURRENT_SCHEMA_VERSION)
        assert reverted == 0

    def test_rollback_target_negative_raises(self, tmp_db_path):
        """target_version < 0 应抛 ValueError。"""
        store = SqliteFeedbackStore(db_path=tmp_db_path)
        with pytest.raises(ValueError):
            store.rollback(target_version=-1)

    def test_rollback_preserves_v1_data(self, populated_db_path):
        """回滚到 v1 不应丢失 feedback 表数据。"""
        db_path, _, _ = populated_db_path
        store = SqliteFeedbackStore(db_path=db_path)

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM feedback")
        before_count = cur.fetchone()[0]
        conn.close()

        store.rollback(target_version=1)

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM feedback")
        after_count = cur.fetchone()[0]
        cur.execute("SELECT document FROM feedback")
        docs = [r[0] for r in cur.fetchall()]
        conn.close()
        assert after_count == before_count
        assert any("深度学习" in d for d in docs)
        assert any("React" in d for d in docs)

    def test_upgrade_after_rollback(self, tmp_db_path):
        """回滚后应能再次前向升级并恢复正常功能。"""
        store = SqliteFeedbackStore(db_path=tmp_db_path)
        store.rollback(target_version=0)
        assert store.get_schema_version() == 0
        applied = store.upgrade()
        assert applied == CURRENT_SCHEMA_VERSION
        assert store.get_schema_version() == CURRENT_SCHEMA_VERSION
        fb_id = store.add_feedback("doc", ["x"], ["x"], [], [])
        assert fb_id.startswith("fb_")


class TestMigrationBackfill:
    """测试数据回填。"""

    def test_backfill_rebuilds_feedback_tag(self, tmp_db_path):
        """回滚到 v1、插入手动数据、升级 v2、backfill 后索引应完整。"""
        store = SqliteFeedbackStore(db_path=tmp_db_path)
        store.rollback(target_version=1)

        conn = sqlite3.connect(tmp_db_path)
        cur = conn.cursor()
        import json
        cur.execute(
            "INSERT INTO feedback VALUES (?,?,?,?,?,?,?,?,?)",
            (
                "fb_manual_1",
                "手动插入的文档",
                json.dumps(["A", "B"], ensure_ascii=False),
                json.dumps(["A"], ensure_ascii=False),
                json.dumps(["B"], ensure_ascii=False),
                json.dumps(["C"], ensure_ascii=False),
                json.dumps(["A", "C"], ensure_ascii=False),
                "2026-01-01T00:00:00",
                "{}",
            ),
        )
        conn.commit()
        conn.close()

        store.upgrade(target_version=2)
        count = store.backfill_tag_table()
        assert count == 1

        accepted = store.get_tags_by_type("accepted")
        added = store.get_tags_by_type("added")
        suggested = store.get_tags_by_type("suggested")
        assert any(r["tag"] == "A" for r in accepted)
        assert any(r["tag"] == "C" for r in added)
        assert any(r["tag"] == "B" for r in suggested)

    def test_backfill_with_v1_schema_returns_zero(self, tmp_db_path):
        """v1 schema 下 backfill 返回 0。"""
        store = SqliteFeedbackStore(db_path=tmp_db_path)
        store.rollback(target_version=1)
        assert store.backfill_tag_table() == 0
