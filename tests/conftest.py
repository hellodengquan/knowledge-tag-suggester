import os
import sys
import tempfile
import shutil
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def tmp_db_path(tmp_path):
    """返回一个临时 SQLite 数据库路径 (文件不存在)。"""
    return str(tmp_path / "test_feedback.db")


@pytest.fixture
def populated_db_path(tmp_path):
    """返回一个已升级到 v2 且有反馈数据的 SQLite 数据库路径。"""
    from feedback_store import SqliteFeedbackStore

    db_path = str(tmp_path / "populated.db")
    store = SqliteFeedbackStore(db_path=db_path)
    assert store.get_schema_version() == 2

    fb1 = store.add_feedback(
        document="深度学习在NLP中的应用研究",
        suggested_tags=["深度学习", "NLP", "数据挖掘"],
        accepted_tags=["深度学习", "NLP"],
        rejected_tags=["数据挖掘"],
        user_added_tags=["自然语言处理"]
    )
    fb2 = store.add_feedback(
        document="React 前端性能优化实践",
        suggested_tags=["React", "前端", "后端"],
        accepted_tags=["React", "前端"],
        rejected_tags=["后端"],
        user_added_tags=["性能优化"]
    )
    return db_path, fb1, fb2


@pytest.fixture
def cli_runner(tmp_path, monkeypatch):
    """返回一个方便调用 CLI 子命令并捕获 stdout/stderr/exit code 的辅助函数。"""
    import io
    from contextlib import redirect_stdout, redirect_stderr

    def _run(cli_args):
        from cli import main

        argv = ["tag-suggester"] + cli_args
        monkeypatch.setattr(sys, "argv", argv)

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        exit_code = 0
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                main()
        except SystemExit as e:
            exit_code = e.code if e.code is not None else 0

        return {
            "exit_code": exit_code,
            "stdout": stdout_capture.getvalue(),
            "stderr": stderr_capture.getvalue(),
        }

    return _run


@pytest.fixture
def sample_training_file(tmp_path):
    """创建一个临时训练数据 JSON 文件。"""
    import json

    data = [
        {"summary": "深度学习在图像识别中的应用", "tags": ["深度学习", "图像识别"]},
        {"summary": "React 前端组件开发", "tags": ["React", "前端开发"]},
    ]
    fpath = tmp_path / "training_docs.json"
    fpath.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(fpath)


@pytest.fixture
def sample_tag_desc_file(tmp_path):
    """创建一个临时标签描述 JSON 文件。"""
    import json

    data = {
        "深度学习": "深度学习是机器学习分支，使用多层神经网络",
        "React": "React 是前端 JavaScript UI 库",
    }
    fpath = tmp_path / "tag_descriptions.json"
    fpath.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return str(fpath)
