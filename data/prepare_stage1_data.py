from __future__ import annotations

import argparse
from pathlib import Path

from utils.io import read_json, write_json, write_jsonl, write_lines
from utils.legal import deduplicate_article_ids, extract_article_number


def unique_qid(base_qid: str, used_qids: set[str]) -> str:
    if base_qid not in used_qids:
        used_qids.add(base_qid)
        return base_qid

    suffix = 1
    while True:
        candidate = f"{base_qid}_{suffix}"
        if candidate not in used_qids:
            used_qids.add(candidate)
            return candidate
        suffix += 1


def normalize_articles(raw_articles: list[dict]) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows: list[dict[str, str]] = []
    seen_article_ids: set[str] = set()

    stats = {
        "raw_article_rows": len(raw_articles),
        "article_name_parse_fallback": 0,
        "duplicate_article_id_rows_skipped": 0,
        "empty_article_content_skipped": 0,
    }

    for idx, item in enumerate(raw_articles):
        name = str(item.get("name", "")).strip()
        content = str(item.get("content", "")).strip()

        article_id = extract_article_number(name)
        if article_id is None:
            fallback_numeric = int(item.get("id", idx)) + 1
            article_id = str(fallback_numeric)
            stats["article_name_parse_fallback"] += 1

        if not content:
            stats["empty_article_content_skipped"] += 1
            continue

        if article_id in seen_article_ids:
            stats["duplicate_article_id_rows_skipped"] += 1
            continue

        seen_article_ids.add(article_id)
        rows.append({"article_id": article_id, "text": content})

    rows.sort(key=lambda x: int(x["article_id"]) if x["article_id"].isdigit() else x["article_id"])
    return rows, stats


def extract_gold_article_ids(conversation_turn: dict) -> list[str]:
    candidates: list[str] = []

    for article_ref in conversation_turn.get("article", []) or []:
        article_id = extract_article_number(str(article_ref))
        if article_id is not None:
            candidates.append(article_id)

    for ctx_item in conversation_turn.get("article_context", []) or []:
        if not isinstance(ctx_item, dict):
            continue
        for key in ctx_item.keys():
            article_id = extract_article_number(str(key))
            if article_id is not None:
                candidates.append(article_id)

    return deduplicate_article_ids(candidates)


def normalize_questions(raw_questions: list[dict], article_id_set: set[str]) -> tuple[list[dict], list[str], dict[str, int]]:
    rows: list[dict] = []
    effective_qids: list[str] = []
    used_qids: set[str] = set()

    stats = {
        "raw_question_rows": len(raw_questions),
        "missing_conversation_rows": 0,
        "empty_question_rows": 0,
        "empty_gold_rows": 0,
        "gold_not_in_corpus_rows": 0,
    }

    for idx, item in enumerate(raw_questions):
        conversation = item.get("conversation") or []
        if not conversation:
            stats["missing_conversation_rows"] += 1
            continue

        turn0 = conversation[0]
        question = str(turn0.get("user", "")).strip()
        if not question:
            stats["empty_question_rows"] += 1

        raw_id = item.get("id", idx + 1)
        qid = unique_qid(f"q{raw_id}", used_qids)

        gold_articles = extract_gold_article_ids(turn0)
        if not gold_articles:
            stats["empty_gold_rows"] += 1

        missing_gold = [aid for aid in gold_articles if aid not in article_id_set]
        if missing_gold:
            stats["gold_not_in_corpus_rows"] += 1

        row = {
            "qid": qid,
            "question": question,
            "gold_articles": gold_articles,
        }
        rows.append(row)

        is_effective = bool(question) and bool(gold_articles) and not missing_gold
        if is_effective:
            effective_qids.append(qid)

    return rows, effective_qids, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare stage-1 retrieval data files.")
    parser.add_argument(
        "--raw_questions",
        type=str,
        default="data/merged_civil_code.json",
        help="Path to raw question json file.",
    )
    parser.add_argument(
        "--raw_articles",
        type=str,
        default="data/civil_code_articles_cleaned.json",
        help="Path to raw article json file.",
    )
    parser.add_argument(
        "--out_questions",
        type=str,
        default="data/questions.jsonl",
        help="Output path for normalized questions jsonl.",
    )
    parser.add_argument(
        "--out_articles",
        type=str,
        default="data/civil_code_articles.jsonl",
        help="Output path for normalized articles jsonl.",
    )
    parser.add_argument(
        "--out_effective_qids",
        type=str,
        default="data/effective_qids.txt",
        help="Output path for effective qids text file.",
    )
    parser.add_argument(
        "--effective_count",
        type=int,
        default=320,
        help="How many effective qids to keep. Use <=0 to keep all effective rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_questions = read_json(args.raw_questions)
    raw_articles = read_json(args.raw_articles)

    normalized_articles, article_stats = normalize_articles(raw_articles)
    article_id_set = {row["article_id"] for row in normalized_articles}

    normalized_questions, effective_qids_all, question_stats = normalize_questions(raw_questions, article_id_set)

    if args.effective_count and args.effective_count > 0:
        effective_qids = effective_qids_all[: args.effective_count]
    else:
        effective_qids = effective_qids_all

    write_jsonl(args.out_articles, normalized_articles)
    write_jsonl(args.out_questions, normalized_questions)
    write_lines(args.out_effective_qids, effective_qids)

    effective_qid_set = set(effective_qids)
    effective_rows = [row for row in normalized_questions if row["qid"] in effective_qid_set]
    write_jsonl("outputs/scores/questions_effective.jsonl", effective_rows)

    audit = {
        "inputs": {
            "raw_questions": str(Path(args.raw_questions)),
            "raw_articles": str(Path(args.raw_articles)),
        },
        "outputs": {
            "questions": str(Path(args.out_questions)),
            "articles": str(Path(args.out_articles)),
            "effective_qids": str(Path(args.out_effective_qids)),
            "questions_effective": "outputs/scores/questions_effective.jsonl",
        },
        "article_stats": article_stats,
        "question_stats": question_stats,
        "effective": {
            "effective_found": len(effective_qids_all),
            "effective_selected": len(effective_qids),
            "requested_effective_count": args.effective_count,
        },
    }

    write_json("outputs/scores/data_audit.json", audit, indent=2)

    print("[prepare_data] done")
    print(f"[prepare_data] normalized articles: {len(normalized_articles)}")
    print(f"[prepare_data] normalized questions: {len(normalized_questions)}")
    print(f"[prepare_data] effective qids selected: {len(effective_qids)}")


if __name__ == "__main__":
    main()
