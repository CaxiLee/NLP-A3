"""
从 CLUECorpus2020 中文维基子集目录中提取文本：每行一个 JSON，取 title 与 text 拼接后写入单一文件。
"""
import argparse
import json
import os
from pathlib import Path


def iter_wiki_files(root: Path):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.startswith("wiki_"):
                yield Path(dirpath) / name


def main():
    parser = argparse.ArgumentParser(description="Extract wiki_zh JSON lines to plain text.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="wiki_zh_2019/wiki_zh",
        help="Root folder containing subfolders AA..AM with wiki_* files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="wikizh.txt",
        help="Output text file path.",
    )
    args = parser.parse_args()
    root = Path(args.input_dir)
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sep = "\n\n<|endoftext|>\n\n"
    n_articles = 0
    n_lines_bad = 0

    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        for fp in iter_wiki_files(root):
            with open(fp, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        n_lines_bad += 1
                        continue
                    title = obj.get("title") or ""
                    text = obj.get("text") or ""
                    block = f"{title}\n{text}"
                    if n_articles > 0:
                        out.write(sep)
                    out.write(block)
                    n_articles += 1

    print(f"Wrote {out_path} ({n_articles} articles, {n_lines_bad} bad JSON lines).")


if __name__ == "__main__":
    main()
