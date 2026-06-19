import os
import sqlite3
import threading
from contextlib import contextmanager
from typing import Optional, Iterator, Any, List, Tuple


DEFAULT_PRAGMAS = {
    "journal_mode": "WAL",
    "foreign_keys": "ON",
    "synchronous": "NORMAL",
    "cache_size": "-20000",
}


class SqliteSession:
    """封装 SQLite 连接、PRAGMA 配置和通用操作。"""

    def __init__(
        self,
        db_path: str,
        pragmas: Optional[dict] = None,
        timeout: float = 5.0,
    ):
        self.db_path = db_path
        self.pragmas = {**DEFAULT_PRAGMAS, **(pragmas or {})}
        self.timeout = timeout
        self._thread_lock = threading.Lock()

    def connect(self) -> sqlite3.Connection:
        """创建新连接并应用 PRAGMA。"""
        parent_dir = os.path.dirname(self.db_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=self.timeout)
        for key, value in self.pragmas.items():
            conn.execute(f"PRAGMA {key} = {value}")
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Cursor]:
        """以事务方式打开一个 cursor，块结束自动 commit；异常时 rollback。"""
        conn = self.connect()
        cursor = conn.cursor()
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def readonly(self) -> Iterator[sqlite3.Cursor]:
        """只读打开 cursor，结束后直接关闭，不做 commit/rollback。"""
        conn = self.connect()
        try:
            yield conn.cursor()
        finally:
            conn.close()

    def execute(self, sql: str, params: Tuple[Any, ...] = ()) -> sqlite3.Cursor:
        """执行单条语句并返回 cursor（调用方负责关闭对应的连接）。"""
        with self._thread_lock:
            conn = self.connect()
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            return cursor

    def executemany(self, sql: str, seq_of_params: List[Tuple[Any, ...]]) -> None:
        with self.transaction() as cursor:
            cursor.executemany(sql, seq_of_params)

    def fetchone(self, sql: str, params: Tuple[Any, ...] = ()) -> Optional[Tuple]:
        with self.readonly() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchone()

    def fetchall(self, sql: str, params: Tuple[Any, ...] = ()) -> List[Tuple]:
        with self.readonly() as cursor:
            cursor.execute(sql, params)
            return cursor.fetchall()

    def table_exists(self, table_name: str) -> bool:
        row = self.fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return row is not None

    def column_exists(self, table_name: str, column_name: str) -> bool:
        if not self.table_exists(table_name):
            return False
        with self.readonly() as cursor:
            cursor.execute(f"PRAGMA table_info({table_name})")
            return any(row[1] == column_name for row in cursor.fetchall())
