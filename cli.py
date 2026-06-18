import argparse
import sys
import os
import json

from tag_suggester import TagSuggester
from feedback_store import create_feedback_store
from tag_explainer import TagExplainer


def cmd_train(args):
    suggester = TagSuggester(
        max_features=args.max_features,
        default_top_k=args.top_k,
        default_threshold=args.threshold
    )

    if not args.input:
        print("错误: 请指定训练数据文件 (--input)", file=sys.stderr)
        sys.exit(1)

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    tags_list = []
    for item in data:
        if args.summary_field and args.summary_field in item:
            text = item[args.summary_field]
        elif "summary" in item:
            text = item["summary"]
        elif "content" in item:
            text = item["content"]
        else:
            text = str(item)

        if args.tags_field and args.tags_field in item:
            tags = item[args.tags_field]
        elif "tags" in item:
            tags = item["tags"]
        else:
            tags = []

        documents.append(text)
        tags_list.append(tags)

    print(f"加载训练数据: {len(documents)} 篇文档")
    suggester.fit(documents, tags_list)
    suggester.save(args.model)
    print(f"模型已保存到: {args.model}")


def cmd_suggest(args):
    suggester = TagSuggester()
    if os.path.exists(args.model):
        suggester.load(args.model)
        print(f"已加载模型: {args.model}")
    else:
        print(f"警告: 模型文件 {args.model} 不存在，使用冷启动模式")

    if args.tag_descriptions:
        with open(args.tag_descriptions, 'r', encoding='utf-8') as f:
            tag_descs = json.load(f)
            for tag, desc in tag_descs.items():
                suggester.add_tag_description(tag, desc)
        print(f"已加载 {len(tag_descs)} 个标签描述")

    if args.document:
        document = args.document
    elif args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            document = f.read()
    else:
        print("错误: 请指定文档内容 (--document) 或文件 (--input-file)", file=sys.stderr)
        sys.exit(1)

    tags, scores = suggester.suggest(
        document,
        top_k=args.top_k if args.top_k is not None else None,
        threshold=args.threshold if args.threshold is not None else None
    )

    print(f"\n推荐标签 (Top {min(args.top_k or suggester.default_top_k, len(tags))}):")
    print("-" * 50)
    for i, (tag, score) in enumerate(zip(tags, scores), 1):
        bar = "█" * int(score * 30)
        print(f"  {i:2d}. {tag:<15} {score:.4f}  {bar}")

    if args.explain:
        explainer = TagExplainer(suggester)
        print(f"\n推荐依据:")
        print("-" * 50)
        explanations = explainer.explain_all(document, tags, top_k_words=args.explain_words)
        for exp in explanations:
            words_str = "、".join([f"'{w}'" for w, s in exp["top_contributing_words"]])
            print(f"  - {exp['tag']}:")
            print(f"    {exp['explanation']}")
            if words_str:
                print(f"    关键词: {words_str}")

    if args.json_output:
        result = {
            "tags": tags,
            "scores": [round(s, 4) for s in scores]
        }
        with open(args.json_output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {args.json_output}")


def cmd_feedback(args):
    store = create_feedback_store(
        backend=args.backend,
        filepath=args.storage,
        db_path=args.storage
    )

    if args.action == "add":
        if not args.document:
            print("错误: 请指定文档内容 (--document)", file=sys.stderr)
            sys.exit(1)

        suggested = args.suggested.split(",") if args.suggested else []
        accepted = args.accepted.split(",") if args.accepted else []
        rejected = args.rejected.split(",") if args.rejected else []
        added = args.added.split(",") if args.added else []

        fb_id = store.add_feedback(
            document=args.document,
            suggested_tags=suggested,
            accepted_tags=accepted,
            rejected_tags=rejected,
            user_added_tags=added
        )
        print(f"反馈已记录，ID: {fb_id}")

    elif args.action == "list":
        records = store.list_feedback(limit=args.limit, offset=args.offset)
        print(f"反馈记录 (共 {len(records)} 条):")
        print("-" * 60)
        for r in records:
            doc_preview = r["document"][:40] + "..." if len(r["document"]) > 40 else r["document"]
            print(f"  [{r['id']}] {r['timestamp']}")
            print(f"      文档: {doc_preview}")
            print(f"      最终标签: {', '.join(r['final_tags'])}")

    elif args.action == "stats":
        stats = store.get_stats()
        print("反馈统计:")
        print("-" * 30)
        for k, v in stats.items():
            print(f"  {k}: {v}")

    elif args.action == "get":
        if not args.id:
            print("错误: 请指定反馈 ID (--id)", file=sys.stderr)
            sys.exit(1)
        record = store.get_feedback(args.id)
        if record:
            print(json.dumps(record, ensure_ascii=False, indent=2))
        else:
            print(f"未找到反馈记录: {args.id}")

    elif args.action == "clear":
        if args.force:
            store.clear()
            print("所有反馈记录已清除")
        else:
            print("警告: 此操作将清除所有反馈记录。使用 --force 确认。")


def cmd_retrain(args):
    store = create_feedback_store(
        backend=args.backend,
        filepath=args.storage,
        db_path=args.storage
    )

    suggester = TagSuggester(
        default_top_k=args.top_k,
        default_threshold=args.threshold
    )

    if os.path.exists(args.model):
        suggester.load(args.model)
        print(f"已加载现有模型: {args.model}")

    docs, tags = store.get_training_data()
    print(f"从反馈存储加载训练数据: {len(docs)} 条")

    if not docs:
        print("没有可用的反馈训练数据")
        sys.exit(0)

    if suggester.is_trained:
        suggester.partial_fit(docs, tags)
    else:
        suggester.fit(docs, tags)

    suggester.save(args.model)
    print(f"模型已更新并保存到: {args.model}")


def main():
    parser = argparse.ArgumentParser(
        prog="tag-suggester",
        description="知识标签推荐器 CLI - 基于 scikit-learn 的多标签推荐工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用样本数据训练模型
  tag-suggester train -i sample_data.json -m model.pkl

  # 为文档推荐标签
  tag-suggester suggest -d "本文研究深度学习在NLP中的应用" -m model.pkl -k 3

  # 推荐并显示依据
  tag-suggester suggest -d "文档内容..." --explain

  # 提交人工反馈
  tag-suggester feedback add -d "文档..." -s "标签1,标签2" -a "标签1" -r "标签2" --added "新标签"

  # 查看反馈统计
  tag-suggester feedback stats

  # 使用 SQLite 后端
  tag-suggester feedback stats --backend sqlite --storage feedback.db

  # 用反馈数据增量训练
  tag-suggester retrain -m model.pkl
        """
    )
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # train 子命令
    train_parser = subparsers.add_parser("train", help="训练标签推荐模型")
    train_parser.add_argument("-i", "--input", required=True, help="训练数据 JSON 文件路径")
    train_parser.add_argument("-m", "--model", default="tag_model.pkl", help="模型输出路径")
    train_parser.add_argument("--summary-field", default=None, help="文档摘要字段名")
    train_parser.add_argument("--tags-field", default=None, help="标签字段名")
    train_parser.add_argument("--max-features", type=int, default=5000, help="TF-IDF 最大特征数")
    train_parser.add_argument("-k", "--top-k", type=int, default=5, help="默认返回标签数量")
    train_parser.add_argument("-t", "--threshold", type=float, default=0.1, help="默认置信度阈值")
    train_parser.set_defaults(func=cmd_train)

    # suggest 子命令
    suggest_parser = subparsers.add_parser("suggest", help="为文档推荐标签")
    suggest_parser.add_argument("-d", "--document", default=None, help="文档摘要文本")
    suggest_parser.add_argument("-f", "--input-file", default=None, help="从文件读取文档内容")
    suggest_parser.add_argument("-m", "--model", default="tag_model.pkl", help="模型文件路径")
    suggest_parser.add_argument("--tag-descriptions", default=None, help="标签描述 JSON 文件 (冷启动用)")
    suggest_parser.add_argument("-k", "--top-k", type=int, default=None, help="返回标签数量 (覆盖默认)")
    suggest_parser.add_argument("-t", "--threshold", type=float, default=None, help="置信度阈值 (覆盖默认)")
    suggest_parser.add_argument("--explain", action="store_true", help="显示推荐依据")
    suggest_parser.add_argument("--explain-words", type=int, default=3, help="解释时显示的关键词数量")
    suggest_parser.add_argument("-o", "--json-output", default=None, help="JSON 结果输出文件")
    suggest_parser.set_defaults(func=cmd_suggest)

    # feedback 子命令
    feedback_parser = subparsers.add_parser("feedback", help="管理人工反馈")
    feedback_parser.add_argument("--backend", choices=["json", "sqlite"], default="json", help="存储后端 (默认: json)")
    feedback_parser.add_argument("-s", "--storage", default=None, help="存储文件路径 (json: .json, sqlite: .db)")
    feedback_sub = feedback_parser.add_subparsers(dest="action", help="反馈操作")

    # feedback add
    fb_add = feedback_sub.add_parser("add", help="添加反馈记录")
    fb_add.add_argument("-d", "--document", required=True, help="文档内容")
    fb_add.add_argument("--suggested", default="", help="推荐的标签 (逗号分隔)")
    fb_add.add_argument("-a", "--accepted", default="", help="接受的标签 (逗号分隔)")
    fb_add.add_argument("-r", "--rejected", default="", help="拒绝的标签 (逗号分隔)")
    fb_add.add_argument("--added", default="", help="用户新增的标签 (逗号分隔)")

    # feedback list
    fb_list = feedback_sub.add_parser("list", help="列出反馈记录")
    fb_list.add_argument("-n", "--limit", type=int, default=None, help="返回记录数")
    fb_list.add_argument("--offset", type=int, default=0, help="偏移量")

    # feedback stats
    feedback_sub.add_parser("stats", help="显示反馈统计")

    # feedback get
    fb_get = feedback_sub.add_parser("get", help="获取单条反馈详情")
    fb_get.add_argument("--id", required=True, help="反馈记录 ID")

    # feedback clear
    fb_clear = feedback_sub.add_parser("clear", help="清除所有反馈记录")
    fb_clear.add_argument("--force", action="store_true", help="确认清除操作")

    feedback_parser.set_defaults(func=cmd_feedback)

    # retrain 子命令
    retrain_parser = subparsers.add_parser("retrain", help="用反馈数据增量训练模型")
    retrain_parser.add_argument("-m", "--model", default="tag_model.pkl", help="模型文件路径")
    retrain_parser.add_argument("--backend", choices=["json", "sqlite"], default="json", help="存储后端")
    retrain_parser.add_argument("-s", "--storage", default=None, help="存储文件路径")
    retrain_parser.add_argument("-k", "--top-k", type=int, default=5, help="默认返回标签数量")
    retrain_parser.add_argument("-t", "--threshold", type=float, default=0.1, help="默认置信度阈值")
    retrain_parser.set_defaults(func=cmd_retrain)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if hasattr(args, 'storage') and args.storage is None:
        if args.backend == "json":
            args.storage = "feedback_data.json"
        else:
            args.storage = "feedback_data.db"

    args.func(args)


if __name__ == "__main__":
    main()
