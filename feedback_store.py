import json
import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from db_session import SqliteSession


SCHEMA_MIGRATIONS = {
    1: {
        "up": [
            '''CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                document TEXT NOT NULL,
                suggested_tags TEXT NOT NULL,
                accepted_tags TEXT NOT NULL,
                rejected_tags TEXT NOT NULL,
                user_added_tags TEXT NOT NULL,
                final_tags TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT NOT NULL
            )'''
        ],
        "down": [
            '''DROP TABLE IF EXISTS feedback'''
        ]
    },
    2: {
        "up": [
            '''CREATE TABLE IF NOT EXISTS feedback_tag (
                feedback_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                tag_type TEXT NOT NULL CHECK(tag_type IN ('suggested', 'accepted', 'rejected', 'added', 'final')),
                PRIMARY KEY (feedback_id, tag, tag_type),
                FOREIGN KEY (feedback_id) REFERENCES feedback(id) ON DELETE CASCADE
            )''',
            '''CREATE INDEX IF NOT EXISTS idx_feedback_tag_type ON feedback_tag(tag_type)''',
            '''CREATE INDEX IF NOT EXISTS idx_feedback_tag_name ON feedback_tag(tag)'''
        ],
        "down": [
            '''DROP INDEX IF EXISTS idx_feedback_tag_name''',
            '''DROP INDEX IF EXISTS idx_feedback_tag_type''',
            '''DROP TABLE IF EXISTS feedback_tag'''
        ]
    },
}

CURRENT_SCHEMA_VERSION = max(SCHEMA_MIGRATIONS.keys())


class BaseFeedbackStore(ABC):
    @abstractmethod
    def add_feedback(
        self,
        document: str,
        suggested_tags: List[str],
        accepted_tags: List[str],
        rejected_tags: List[str],
        user_added_tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        pass

    @abstractmethod
    def get_feedback(self, feedback_id: str) -> Optional[Dict]:
        pass

    @abstractmethod
    def list_feedback(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict]:
        pass

    @abstractmethod
    def get_training_data(self) -> tuple:
        pass

    @abstractmethod
    def get_stats(self) -> Dict:
        pass

    @abstractmethod
    def clear(self):
        pass


class JsonFeedbackStore(BaseFeedbackStore):
    def __init__(self, filepath: str = "feedback_data.json"):
        self.filepath = filepath
        self.feedback_records: List[Dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, 'r', encoding='utf-8') as f:
                self.feedback_records = json.load(f)

    def _save(self):
        with open(self.filepath, 'w', encoding='utf-8') as f:
            json.dump(self.feedback_records, f, ensure_ascii=False, indent=2)

    def add_feedback(
        self,
        document: str,
        suggested_tags: List[str],
        accepted_tags: List[str],
        rejected_tags: List[str],
        user_added_tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        record = {
            "id": f"fb_{int(datetime.now().timestamp() * 1000)}",
            "document": document,
            "suggested_tags": suggested_tags,
            "accepted_tags": accepted_tags,
            "rejected_tags": rejected_tags,
            "user_added_tags": user_added_tags or [],
            "final_tags": accepted_tags + (user_added_tags or []),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.feedback_records.append(record)
        self._save()
        return record["id"]

    def get_feedback(self, feedback_id: str) -> Optional[Dict]:
        for record in self.feedback_records:
            if record["id"] == feedback_id:
                return record
        return None

    def list_feedback(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict]:
        records = self.feedback_records[offset:]
        if limit:
            records = records[:limit]
        return list(reversed(records))

    def get_training_data(self) -> tuple:
        documents = []
        tags_list = []
        for record in self.feedback_records:
            if record["final_tags"]:
                documents.append(record["document"])
                tags_list.append(record["final_tags"])
        return documents, tags_list

    def get_stats(self) -> Dict:
        total = len(self.feedback_records)
        if total == 0:
            return {"total": 0, "avg_accepted": 0, "avg_rejected": 0, "avg_added": 0, "acceptance_rate": 0}

        total_accepted = sum(len(r["accepted_tags"]) for r in self.feedback_records)
        total_rejected = sum(len(r["rejected_tags"]) for r in self.feedback_records)
        total_added = sum(len(r["user_added_tags"]) for r in self.feedback_records)
        total_suggested = sum(len(r["suggested_tags"]) for r in self.feedback_records)

        acceptance_rate = total_accepted / total_suggested if total_suggested > 0 else 0

        return {
            "total": total,
            "avg_accepted": round(total_accepted / total, 2),
            "avg_rejected": round(total_rejected / total, 2),
            "avg_added": round(total_added / total, 2),
            "acceptance_rate": round(acceptance_rate, 2)
        }

    def clear(self):
        self.feedback_records = []
        self._save()


class SqliteFeedbackStore(BaseFeedbackStore):
    def __init__(self, db_path: str = "feedback_data.db", pragmas: Optional[dict] = None):
        self.db_path = db_path
        self.session = SqliteSession(db_path=db_path, pragmas=pragmas)
        self._init_db()

    def _init_db(self):
        with self.session.transaction() as cursor:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                )
            ''')
        self._run_migrations()

    def _run_migrations(self):
        with self.session.readonly() as cursor:
            cursor.execute("SELECT MAX(version) FROM schema_version")
            row = cursor.fetchone()
        current_version = row[0] if row[0] is not None else 0

        for version in sorted(SCHEMA_MIGRATIONS.keys()):
            if version > current_version:
                migration = SCHEMA_MIGRATIONS[version]
                with self.session.transaction() as cursor:
                    for sql in migration["up"]:
                        cursor.execute(sql)
                    cursor.execute(
                        "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                        (version, datetime.now().isoformat())
                    )

    def _run_rollback(self, target_version: int):
        with self.session.readonly() as cursor:
            cursor.execute("SELECT version FROM schema_version ORDER BY version DESC")
            applied_versions = [r[0] for r in cursor.fetchall()]

        for version in applied_versions:
            if version > target_version:
                migration = SCHEMA_MIGRATIONS.get(version)
                if migration and "down" in migration:
                    with self.session.transaction() as cursor:
                        for sql in migration["down"]:
                            cursor.execute(sql)
                        cursor.execute("DELETE FROM schema_version WHERE version = ?", (version,))

    def get_schema_version(self) -> int:
        row = self.session.fetchone("SELECT MAX(version) FROM schema_version")
        return row[0] if row and row[0] is not None else 0

    def rollback(self, target_version: int = 0) -> int:
        if target_version < 0:
            raise ValueError("target_version 必须 >= 0")
        if target_version >= self.get_schema_version():
            return 0
        prev = self.get_schema_version()
        self._run_rollback(target_version)
        new_version = self.get_schema_version()
        return prev - new_version

    def upgrade(self, target_version: Optional[int] = None) -> int:
        if target_version is None:
            target_version = CURRENT_SCHEMA_VERSION
        prev = self.get_schema_version()
        self._run_migrations()
        new_version = self.get_schema_version()
        return new_version - prev

    @staticmethod
    def _tags_to_json(tags: List[str]) -> str:
        return json.dumps(tags, ensure_ascii=False)

    @staticmethod
    def _json_to_tags(json_str: str) -> List[str]:
        return json.loads(json_str) if json_str else []

    def _row_to_dict(self, row: Tuple) -> Dict:
        return {
            "id": row[0],
            "document": row[1],
            "suggested_tags": self._json_to_tags(row[2]),
            "accepted_tags": self._json_to_tags(row[3]),
            "rejected_tags": self._json_to_tags(row[4]),
            "user_added_tags": self._json_to_tags(row[5]),
            "final_tags": self._json_to_tags(row[6]),
            "timestamp": row[7],
            "metadata": json.loads(row[8]) if row[8] else {}
        }

    def _sync_tags_to_tag_table(self, cursor, feedback_id: str, tags_by_type: Dict[str, List[str]]):
        cursor.execute("DELETE FROM feedback_tag WHERE feedback_id = ?", (feedback_id,))
        for tag_type, tags in tags_by_type.items():
            for tag in tags:
                cursor.execute(
                    "INSERT INTO feedback_tag (feedback_id, tag, tag_type) VALUES (?, ?, ?)",
                    (feedback_id, tag, tag_type)
                )

    def add_feedback(
        self,
        document: str,
        suggested_tags: List[str],
        accepted_tags: List[str],
        rejected_tags: List[str],
        user_added_tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        user_added = user_added_tags or []
        final_tags = accepted_tags + user_added
        feedback_id = f"fb_{int(datetime.now().timestamp() * 1000)}"
        timestamp = datetime.now().isoformat()

        with self.session.transaction() as cursor:
            cursor.execute('''
                INSERT INTO feedback (
                    id, document, suggested_tags, accepted_tags, rejected_tags,
                    user_added_tags, final_tags, timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback_id,
                document,
                self._tags_to_json(suggested_tags),
                self._tags_to_json(accepted_tags),
                self._tags_to_json(rejected_tags),
                self._tags_to_json(user_added),
                self._tags_to_json(final_tags),
                timestamp,
                json.dumps(metadata or {}, ensure_ascii=False)
            ))
            if self.get_schema_version() >= 2:
                self._sync_tags_to_tag_table(cursor, feedback_id, {
                    "suggested": suggested_tags,
                    "accepted": accepted_tags,
                    "rejected": rejected_tags,
                    "added": user_added,
                    "final": final_tags,
                })
        return feedback_id

    def get_feedback(self, feedback_id: str) -> Optional[Dict]:
        row = self.session.fetchone('SELECT * FROM feedback WHERE id = ?', (feedback_id,))
        if row:
            return self._row_to_dict(row)
        return None

    def list_feedback(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict]:
        query = 'SELECT * FROM feedback ORDER BY timestamp DESC'
        params: Tuple = ()
        if limit is not None:
            query += ' LIMIT ? OFFSET ?'
            params = (limit, offset)
        with self.session.readonly() as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def get_training_data(self) -> tuple:
        rows = self.session.fetchall('SELECT document, final_tags FROM feedback WHERE final_tags != "[]"')
        documents = []
        tags_list = []
        for doc, final_tags_json in rows:
            documents.append(doc)
            tags_list.append(self._json_to_tags(final_tags_json))
        return documents, tags_list

    def get_stats(self) -> Dict:
        total_row = self.session.fetchone('SELECT COUNT(*) FROM feedback')
        total = total_row[0] if total_row else 0
        if total == 0:
            return {"total": 0, "avg_accepted": 0, "avg_rejected": 0, "avg_added": 0, "acceptance_rate": 0}

        rows = self.session.fetchall(
            'SELECT suggested_tags, accepted_tags, rejected_tags, user_added_tags FROM feedback'
        )

        total_accepted = 0
        total_rejected = 0
        total_added = 0
        total_suggested = 0
        for row in rows:
            total_suggested += len(self._json_to_tags(row[0]))
            total_accepted += len(self._json_to_tags(row[1]))
            total_rejected += len(self._json_to_tags(row[2]))
            total_added += len(self._json_to_tags(row[3]))

        acceptance_rate = total_accepted / total_suggested if total_suggested > 0 else 0

        return {
            "total": total,
            "avg_accepted": round(total_accepted / total, 2),
            "avg_rejected": round(total_rejected / total, 2),
            "avg_added": round(total_added / total, 2),
            "acceptance_rate": round(acceptance_rate, 2)
        }

    def clear(self):
        with self.session.transaction() as cursor:
            cursor.execute('DELETE FROM feedback')

    def get_tags_by_type(self, tag_type: str) -> List[Dict]:
        if self.get_schema_version() < 2:
            return []
        rows = self.session.fetchall('''
            SELECT f.id, f.document, ft.tag
            FROM feedback_tag ft
            JOIN feedback f ON ft.feedback_id = f.id
            WHERE ft.tag_type = ?
            ORDER BY f.timestamp DESC
        ''', (tag_type,))
        return [{"feedback_id": r[0], "document": r[1], "tag": r[2]} for r in rows]

    def backfill_tag_table(self) -> int:
        if self.get_schema_version() < 2:
            return 0
        rows = self.session.fetchall(
            'SELECT id, suggested_tags, accepted_tags, rejected_tags, user_added_tags, final_tags FROM feedback'
        )
        with self.session.transaction() as cursor:
            for row in rows:
                fb_id = row[0]
                self._sync_tags_to_tag_table(cursor, fb_id, {
                    "suggested": self._json_to_tags(row[1]),
                    "accepted": self._json_to_tags(row[2]),
                    "rejected": self._json_to_tags(row[3]),
                    "added": self._json_to_tags(row[4]),
                    "final": self._json_to_tags(row[5]),
                })
        return len(rows)


def create_feedback_store(backend: str = "json", **kwargs) -> BaseFeedbackStore:
    if backend == "json":
        filepath = kwargs.get("filepath", "feedback_data.json")
        return JsonFeedbackStore(filepath=filepath)
    elif backend == "sqlite":
        db_path = kwargs.get("db_path", "feedback_data.db")
        pragmas = kwargs.get("pragmas", None)
        return SqliteFeedbackStore(db_path=db_path, pragmas=pragmas)
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: json, sqlite")


FeedbackStore = JsonFeedbackStore
