import argparse
import sys
import os
import json

from tag_suggester import TagSuggester
from feedback_store import create_feedback_store, SqliteFeedbackStore
from tag_explainer import TagExplainer


class ValidationError(Exception):
    pass


def validate_int_range(value, name, min_val=None, max_val=None):
    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} 不能小于 {min_val}，当前值: {value}")
    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} 不能大于 {max_val}，当前值: {value}")


def validate_float_range(value, name, min_val=None, max_val=None):
    if min_val is not None and value < min_val:
        raise ValidationError(f"{name} 不能小于 {min_val}，当前值: {value}")
    if max_val is not None and value > max_val:
        raise ValidationError(f"{name} 不能大于 {max_val}，当前值: {value}")


def validate_file_exists(path, name):
    if not os.path.exists(path):
        raise ValidationError(f"{name} 文件不存在: {path}")


def cmd_train(args):
    try:
        validate_file_exists(args.input, "--input")
        validate_int_range(args.max_features, "--max-features", min_val=1, max_val=100000)
        validate_int_range(args.top_k, "--top-k", min_val=1, max_val=100)
        validate_float_range(args.threshold, "--threshold", min_val=0.0, max_val=1.0)
        if bool(args.summary_field) != bool(args.tags_field):
            raise ValidationError(
                "--summary-field 和 --tags-field 必须同时指定或都不指定"
            )
    except ValidationError as e:
        print(f"参数错误: {e}", file=sys.stderr)
        sys.exit(1)

    suggester = TagSuggester(
        max_features=args.max_features,
        default_top_k=args.top_k,
        default_threshold=args.threshold
    )

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
    try:
        if args.top_k is not None:
            validate_int_range(args.top_k, "--top-k", min_val=1, max_val=100)
        if args.threshold is not None:
            validate_float_range(args.threshold, "--threshold", min_val=0.0, max_val=1.0)
        validate_int_range(args.explain_words, "--explain-words", min_val=1, max_val=20)
        if args.tag_descriptions:
            validate_file_exists(args.tag_descriptions, "--tag-descriptions")
        if args.input_file:
            validate_file_exists(args.input_file, "--input-file")
        if args.document and args.input_file:
            raise ValidationError("--document 和 --input-file 互斥，只能指定其中一个")
        if not args.document and not args.input_file:
            raise ValidationError("--document 或 --input-file 必须至少指定一个")
        if not os.path.exists(args.model) and not args.tag_descriptions:
            raise ValidationError(
                f"模型文件 {args.model} 不存在，冷启动模式下需要通过 --tag-descriptions 指定标签描述"
            )
        if args.explain_words != 3 and not args.explain:
            raise ValidationError("--explain-words 只能在 --explain 开启时使用")
    except ValidationError as e:
        print(f"参数错误: {e}", file=sys.stderr)
        sys.exit(1)

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

    effective_k = args.top_k if args.top_k is not None else suggester.default_top_k
    print(f"\n推荐标签 (Top {min(effective_k, len(tags))}):")
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
        try:
            if args.limit is not None:
                validate_int_range(args.limit, "--limit", min_val=1, max_val=10000)
            validate_int_range(args.offset, "--offset", min_val=0)
        except ValidationError as e:
            print(f"参数错误: {e}", file=sys.stderr)
            sys.exit(1)

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


def cmd_migrate(args):
    if args.action != "status":
        try:
            if not os.path.exists(args.db_path):
                raise ValidationError(f"--db-path 文件不存在: {args.db_path}")
        except ValidationError as e:
            print(f"参数错误: {e}", file=sys.stderr)
            sys.exit(1)

    store = SqliteFeedbackStore(db_path=args.db_path)
    current_version = store.get_schema_version()

    if args.action == "status":
        from feedback_store import CURRENT_SCHEMA_VERSION
        print(f"数据库: {args.db_path}")
        print(f"当前 schema 版本: {current_version}")
        print(f"最新 schema 版本: {max(CURRENT_SCHEMA_VERSION, current_version)}")

    elif args.action == "upgrade":
        from feedback_store import CURRENT_SCHEMA_VERSION
        target = args.target_version if args.target_version is not None else CURRENT_SCHEMA_VERSION
        if target < current_version:
            print(f"目标版本 {target} 小于当前版本 {current_version}，请使用 rollback", file=sys.stderr)
            sys.exit(1)
        applied = store.upgrade(target_version=target)
        print(f"已升级 {applied} 个版本，当前 schema 版本: {store.get_schema_version()}")

    elif args.action == "rollback":
        if args.target_version is None:
            print("错误: rollback 需要 --target-version 指定目标版本", file=sys.stderr)
            sys.exit(1)
        try:
            validate_int_range(args.target_version, "--target-version", min_val=0)
        except ValidationError as e:
            print(f"参数错误: {e}", file=sys.stderr)
            sys.exit(1)
        reverted = store.rollback(target_version=args.target_version)
        print(f"已回滚 {reverted} 个版本，当前 schema 版本: {store.get_schema_version()}")

    elif args.action == "backfill":
        if current_version < 2:
            print("错误: schema 版本 < 2，请先运行 upgrade", file=sys.stderr)
            sys.exit(1)
        count = store.backfill_tag_table()
        print(f"已回填 {count} 条反馈的标签到 feedback_tag 表")

    else:
        from feedback_store import CURRENT_SCHEMA_VERSION
        print(f"当前 schema 版本: {current_version}")
        print(f"迁移完成，数据库已是最新版本")


def cmd_retrain(args):
    try:
        validate_int_range(args.top_k, "--top-k", min_val=1, max_val=100)
        validate_float_range(args.threshold, "--threshold", min_val=0.0, max_val=1.0)
    except ValidationError as e:
        print(f"参数错误: {e}", file=sys.stderr)
        sys.exit(1)

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
        description="知识标签推荐器 — 基于 TF-IDF + 多标签分类的文档标签推荐工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
可用命令:
  train      使用标注数据训练推荐模型
  suggest    为文档摘要推荐知识标签
  feedback   管理人工反馈记录 (添加/查看/统计/清除)
  migrate    管理 SQLite schema 版本与数据回填
  retrain    使用反馈数据增量更新模型

示例:
  # 训练模型
  tag-suggester train -i docs.json -m model.pkl -k 5 -t 0.05

  # 推荐标签并显示依据
  tag-suggester suggest -d "深度学习在NLP中的应用" -m model.pkl --explain

  # 使用 SQLite 后端提交反馈
  tag-suggester feedback --backend sqlite -s data.db add \\
      -d "文档摘要" --suggested "AI,ML" --accepted "AI" --added "NLP"

  # 查看 SQLite schema 版本
  tag-suggester migrate --db-path data.db status

  # 回填 feedback_tag 表
  tag-suggester migrate --db-path data.db backfill

  # 用反馈数据增量训练
  tag-suggester retrain -m model.pkl --backend sqlite -s data.db
"""
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── train ──
    train_parser = subparsers.add_parser(
        "train",
        help="训练标签推荐模型",
        description="从标注数据训练 TF-IDF + 多标签分类模型，输出 pickle 文件。",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    train_parser.add_argument(
        "-i", "--input", required=True,
        help="训练数据 JSON 文件 (每项含 summary + tags 字段)"
    )
    train_parser.add_argument(
        "-m", "--model", default="tag_model.pkl",
        help="模型输出路径 (默认: tag_model.pkl)"
    )
    train_parser.add_argument(
        "--summary-field", default=None,
        help="文档摘要字段名 (默认依次尝试 summary/content)"
    )
    train_parser.add_argument(
        "--tags-field", default=None,
        help="标签字段名 (默认: tags)"
    )
    train_parser.add_argument(
        "--max-features", type=int, default=5000,
        help="TF-IDF 最大特征数 (范围: 1-100000, 默认: 5000)"
    )
    train_parser.add_argument(
        "-k", "--top-k", type=int, default=5,
        help="默认返回标签数量 (范围: 1-100, 默认: 5)"
    )
    train_parser.add_argument(
        "-t", "--threshold", type=float, default=0.1,
        help="默认置信度阈值 (范围: 0.0-1.0, 默认: 0.1)"
    )
    train_parser.set_defaults(func=cmd_train)

    # ── suggest ──
    suggest_parser = subparsers.add_parser(
        "suggest",
        help="为文档推荐标签",
        description="加载训练好的模型，对文档摘要进行标签推荐。支持显示推荐依据。",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    suggest_parser.add_argument(
        "-d", "--document", default=None,
        help="文档摘要文本"
    )
    suggest_parser.add_argument(
        "-f", "--input-file", default=None,
        help="从文件读取文档内容"
    )
    suggest_parser.add_argument(
        "-m", "--model", default="tag_model.pkl",
        help="模型文件路径 (默认: tag_model.pkl)"
    )
    suggest_parser.add_argument(
        "--tag-descriptions", default=None,
        help="标签描述 JSON 文件 (冷启动用)"
    )
    suggest_parser.add_argument(
        "-k", "--top-k", type=int, default=None,
        help="返回标签数量 (范围: 1-100, 不指定则使用模型默认值)"
    )
    suggest_parser.add_argument(
        "-t", "--threshold", type=float, default=None,
        help="置信度阈值 (范围: 0.0-1.0, 不指定则使用模型默认值)"
    )
    suggest_parser.add_argument(
        "--explain", action="store_true",
        help="显示每个推荐标签的依据"
    )
    suggest_parser.add_argument(
        "--explain-words", type=int, default=3,
        help="解释时显示的关键词数量 (范围: 1-20, 默认: 3)"
    )
    suggest_parser.add_argument(
        "-o", "--json-output", default=None,
        help="JSON 结果输出文件路径"
    )
    suggest_parser.set_defaults(func=cmd_suggest)

    # ── feedback ──
    feedback_parser = subparsers.add_parser(
        "feedback",
        help="管理人工反馈记录",
        description="对推荐结果提交人工反馈，支持 JSON 和 SQLite 两种存储后端。",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    feedback_parser.add_argument(
        "--backend", choices=["json", "sqlite"], default="json",
        help="存储后端 (默认: json)"
    )
    feedback_parser.add_argument(
        "-s", "--storage", default=None,
        help="存储文件路径 (json: *.json, sqlite: *.db)"
    )
    feedback_sub = feedback_parser.add_subparsers(dest="action", metavar="ACTION")

    fb_add = feedback_sub.add_parser(
        "add",
        help="添加反馈记录",
        description="对一条推荐结果提交人工反馈：接受/拒绝/新增标签。"
    )
    fb_add.add_argument("-d", "--document", required=True, help="文档内容")
    fb_add.add_argument("--suggested", default="", help="推荐的标签 (逗号分隔)")
    fb_add.add_argument("-a", "--accepted", default="", help="接受的标签 (逗号分隔)")
    fb_add.add_argument("-r", "--rejected", default="", help="拒绝的标签 (逗号分隔)")
    fb_add.add_argument("--added", default="", help="用户新增的标签 (逗号分隔)")

    fb_list = feedback_sub.add_parser(
        "list",
        help="列出反馈记录",
        description="分页列出已存储的反馈记录。"
    )
    fb_list.add_argument(
        "-n", "--limit", type=int, default=None,
        help="返回记录数 (范围: 1-10000)"
    )
    fb_list.add_argument(
        "--offset", type=int, default=0,
        help="偏移量 (范围: ≥0, 默认: 0)"
    )

    feedback_sub.add_parser("stats", help="显示反馈统计")

    fb_get = feedback_sub.add_parser("get", help="获取单条反馈详情")
    fb_get.add_argument("--id", required=True, help="反馈记录 ID")

    fb_clear = feedback_sub.add_parser("clear", help="清除所有反馈记录")
    fb_clear.add_argument("--force", action="store_true", help="确认清除操作")

    feedback_parser.set_defaults(func=cmd_feedback)

    # ── migrate ──
    migrate_parser = subparsers.add_parser(
        "migrate",
        help="管理 SQLite schema 版本与数据回填",
        description="查看 SQLite 数据库的 schema 版本、执行前向升级、回滚、回填标签索引表。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
子命令:
  status    查看 schema 版本信息
  upgrade   前向升级到最新或指定版本 (--target-version)
  rollback  回滚到指定版本 (需要 --target-version)
  backfill  将 feedback 表中的标签数据回填到 feedback_tag 索引表

Schema 版本历史:
  v1  初始 feedback 表 (JSON 列存储标签)
  v2  新增 feedback_tag 表 (标签索引，支持按类型查询)
"""
    )
    migrate_parser.add_argument(
        "--db-path", default="feedback_data.db",
        help="SQLite 数据库路径 (默认: feedback_data.db)"
    )
    migrate_sub = migrate_parser.add_subparsers(dest="action", metavar="ACTION")

    migrate_sub.add_parser("status", help="查看当前 schema 版本")

    up_p = migrate_sub.add_parser("upgrade", help="前向升级 schema 到最新或指定版本")
    up_p.add_argument("--target-version", type=int, default=None,
                      help="目标版本 (默认: 最新版本)")

    rb_p = migrate_sub.add_parser("rollback", help="回滚 schema 到指定版本")
    rb_p.add_argument("--target-version", type=int, required=True,
                      help="目标版本 (必须指定，0 表示完全回滚)")

    migrate_sub.add_parser("backfill", help="回填 feedback_tag 索引表")

    migrate_parser.set_defaults(func=cmd_migrate)

    # ── retrain ──
    retrain_parser = subparsers.add_parser(
        "retrain",
        help="用反馈数据增量训练模型",
        description="从反馈存储中加载标注数据，对现有模型进行增量更新。",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    retrain_parser.add_argument(
        "-m", "--model", default="tag_model.pkl",
        help="模型文件路径 (默认: tag_model.pkl)"
    )
    retrain_parser.add_argument(
        "--backend", choices=["json", "sqlite"], default="json",
        help="存储后端 (默认: json)"
    )
    retrain_parser.add_argument(
        "-s", "--storage", default=None,
        help="存储文件路径"
    )
    retrain_parser.add_argument(
        "-k", "--top-k", type=int, default=5,
        help="默认返回标签数量 (范围: 1-100, 默认: 5)"
    )
    retrain_parser.add_argument(
        "-t", "--threshold", type=float, default=0.1,
        help="默认置信度阈值 (范围: 0.0-1.0, 默认: 0.1)"
    )
    retrain_parser.set_defaults(func=cmd_retrain)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if hasattr(args, 'storage') and args.storage is None:
        if hasattr(args, 'backend'):
            if args.backend == "json":
                args.storage = "feedback_data.json"
            else:
                args.storage = "feedback_data.db"

    args.func(args)


if __name__ == "__main__":
    main()
