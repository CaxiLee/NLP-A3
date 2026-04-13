"""统计文本在指定 BPE tokenizer 下的 token 总数（分块编码，降低内存占用）。"""
import argparse

from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, default="wikizh_tokenizer.json")
    parser.add_argument("--text", type=str, default="wikizh.txt")
    parser.add_argument(
        "--chunk_chars",
        type=int,
        default=5_000_000,
        help="每次读入的字符数；块边界处 token 计数可能有微小误差。",
    )
    args = parser.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    total = 0
    with open(args.text, "r", encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(args.chunk_chars)
            if not chunk:
                break
            total += len(tok.encode(chunk).ids)
    print(total)


if __name__ == "__main__":
    main()
