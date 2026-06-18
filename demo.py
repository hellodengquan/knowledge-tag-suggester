from tag_suggester import TagSuggester
from feedback_store import FeedbackStore
from tag_explainer import TagExplainer
from sample_data import SAMPLE_DOCUMENTS, TAG_DESCRIPTIONS
import json


def print_separator(title=""):
    line = "=" * 60
    if title:
        print(f"\n{line}")
        print(f"  {title}")
        print(line)
    else:
        print(f"\n{line}")


def main():
    print_separator("知识标签推荐器 - 演示")

    print("\n[1/5] 初始化推荐器...")
    suggester = TagSuggester(max_features=1000)

    for tag, desc in TAG_DESCRIPTIONS.items():
        suggester.add_tag_description(tag, desc)

    print(f"  已加载 {len(TAG_DESCRIPTIONS)} 个标签描述")

    print("\n[2/5] 训练模型...")
    documents = [doc["summary"] for doc in SAMPLE_DOCUMENTS]
    tags_list = [doc["tags"] for doc in SAMPLE_DOCUMENTS]
    suggester.fit(documents, tags_list)
    print(f"  训练完成，共 {len(documents)} 篇文档")

    feedback_store = FeedbackStore("demo_feedback.json")
    explainer = TagExplainer(suggester)

    print("\n[3/5] 标签推荐演示...")

    test_docs = [
        {
            "title": "测试文档1 - 深度学习与NLP",
            "content": "本文研究了基于Transformer的预训练语言模型在文本分类任务中的表现，通过微调BERT模型实现了较高的准确率。"
        },
        {
            "title": "测试文档2 - 前端开发",
            "content": "使用React和Vue构建现代化Web应用的对比分析，包括组件化开发、状态管理方案和性能优化策略的详细比较。"
        },
        {
            "title": "测试文档3 - 数据库与后端",
            "content": "MySQL数据库索引优化实战，详解B+树索引原理、慢查询分析方法以及分库分表架构的设计思路。"
        }
    ]

    for i, test_doc in enumerate(test_docs, 1):
        print_separator(f"测试文档 {i}: {test_doc['title']}")
        print(f"内容摘要: {test_doc['content']}")

        suggested_tags, scores = suggester.suggest(test_doc["content"], top_k=5, threshold=0.1)

        print(f"\n推荐标签 (Top 5):")
        for j, (tag, score) in enumerate(zip(suggested_tags, scores), 1):
            print(f"  {j}. {tag} - 置信度: {score:.4f}")

        print(f"\n推荐依据:")
        explanations = explainer.explain_all(test_doc["content"], suggested_tags, top_k_words=3)
        for exp in explanations:
            words_str = "、".join([f"'{w}'" for w, s in exp["top_contributing_words"]])
            print(f"  - {exp['tag']}: {exp['explanation']}")
            if exp["top_contributing_words"]:
                print(f"    关键词贡献: {words_str}")

        print(f"\n文档关键词 (Top 8):")
        keywords = explainer.get_document_keywords(test_doc["content"], top_k=8)
        for j, (word, score) in enumerate(keywords, 1):
            print(f"  {j}. {word} (TF-IDF: {score:.4f})")

    print_separator("[4/5] 人工反馈演示")

    print("\n模拟用户对测试文档1的反馈...")
    test_content = test_docs[0]["content"]
    suggested_tags, _ = suggester.suggest(test_content, top_k=5)

    accepted = suggested_tags[:2]
    rejected = suggested_tags[2:4]
    user_added = ["预训练模型"]

    print(f"  推荐标签: {suggested_tags}")
    print(f"  用户接受: {accepted}")
    print(f"  用户拒绝: {rejected}")
    print(f"  用户新增: {user_added}")

    fb_id = feedback_store.add_feedback(
        document=test_content,
        suggested_tags=suggested_tags,
        accepted_tags=accepted,
        rejected_tags=rejected,
        user_added_tags=user_added,
        metadata={"source": "demo", "doc_title": test_docs[0]["title"]}
    )
    print(f"  反馈记录ID: {fb_id}")

    print("\n反馈统计:")
    stats = feedback_store.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print_separator("[5/5] 增量学习演示")

    print("\n使用反馈数据进行增量学习...")
    fb_docs, fb_tags = feedback_store.get_training_data()
    print(f"  反馈数据量: {len(fb_docs)} 条")

    if fb_docs:
        suggester.partial_fit(fb_docs, fb_tags)
        print("  增量学习完成！")

    print("\n" + "=" * 60)
    print("  演示完成！")
    print("=" * 60)

    print("\n模块说明:")
    print("  tag_suggester.py  - 标签推荐器核心 (TF-IDF + 多标签分类)")
    print("  feedback_store.py - 人工反馈存储与管理")
    print("  tag_explainer.py  - 推荐依据解释")
    print("  sample_data.py    - 示例数据")
    print("  demo.py           - 演示脚本")

    print("\n文件生成:")
    print("  demo_feedback.json - 反馈数据存储文件")


if __name__ == "__main__":
    main()
