from sample_data import SAMPLE_DOCUMENTS, TAG_DESCRIPTIONS
import json

with open("training_docs.json", "w", encoding="utf-8") as f:
    json.dump(SAMPLE_DOCUMENTS, f, ensure_ascii=False, indent=2)

with open("tag_descriptions.json", "w", encoding="utf-8") as f:
    json.dump(TAG_DESCRIPTIONS, f, ensure_ascii=False, indent=2)

print("已生成 training_docs.json 和 tag_descriptions.json")
