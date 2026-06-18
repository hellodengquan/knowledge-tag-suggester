import json
import os
from datetime import datetime
from typing import List, Dict, Optional


class FeedbackStore:
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
            return {"total": 0, "avg_accepted": 0, "avg_rejected": 0, "avg_added": 0}

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
