import json
import os
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Optional


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
    def __init__(self, db_path: str = "feedback_data.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                document TEXT NOT NULL,
                suggested_tags TEXT NOT NULL,
                accepted_tags TEXT NOT NULL,
                rejected_tags TEXT NOT NULL,
                user_added_tags TEXT NOT NULL,
                final_tags TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def _tags_to_json(self, tags: List[str]) -> str:
        return json.dumps(tags, ensure_ascii=False)

    def _json_to_tags(self, json_str: str) -> List[str]:
        return json.loads(json_str) if json_str else []

    def _row_to_dict(self, row) -> Dict:
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

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
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
        conn.commit()
        conn.close()
        return feedback_id

    def get_feedback(self, feedback_id: str) -> Optional[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM feedback WHERE id = ?', (feedback_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return self._row_to_dict(row)
        return None

    def list_feedback(self, limit: Optional[int] = None, offset: int = 0) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = 'SELECT * FROM feedback ORDER BY timestamp DESC'
        params = []
        if limit is not None:
            query += ' LIMIT ? OFFSET ?'
            params.extend([limit, offset])
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return [self._row_to_dict(row) for row in rows]

    def get_training_data(self) -> tuple:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT document, final_tags FROM feedback WHERE final_tags != "[]"')
        rows = cursor.fetchall()
        conn.close()
        documents = []
        tags_list = []
        for doc, final_tags_json in rows:
            documents.append(doc)
            tags_list.append(self._json_to_tags(final_tags_json))
        return documents, tags_list

    def get_stats(self) -> Dict:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM feedback')
        total = cursor.fetchone()[0]
        if total == 0:
            conn.close()
            return {"total": 0, "avg_accepted": 0, "avg_rejected": 0, "avg_added": 0, "acceptance_rate": 0}

        cursor.execute('SELECT suggested_tags, accepted_tags, rejected_tags, user_added_tags FROM feedback')
        rows = cursor.fetchall()
        conn.close()

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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM feedback')
        conn.commit()
        conn.close()


def create_feedback_store(backend: str = "json", **kwargs) -> BaseFeedbackStore:
    if backend == "json":
        filepath = kwargs.get("filepath", "feedback_data.json")
        return JsonFeedbackStore(filepath=filepath)
    elif backend == "sqlite":
        db_path = kwargs.get("db_path", "feedback_data.db")
        return SqliteFeedbackStore(db_path=db_path)
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: json, sqlite")


FeedbackStore = JsonFeedbackStore
