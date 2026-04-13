import argparse
import os
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def iter_wiki_files(root: Path):
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if name.startswith("wiki_"):
                yield Path(dirpath) / name


parser = argparse.ArgumentParser(
    description="Train a BPE tokenizer on Chinese text data from scratch."
)
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Plain .txt file, or a wiki_zh directory (wiki_* JSON line files under wiki_zh/...).",
)
parser.add_argument(
    "--vocab_size", type=int, default=52000, help="Vocabulary size."
)
parser.add_argument(
    "--pre_tokenizer",
    type=str,
    choices=["Whitespace", "ByteLevel"],
    default="Whitespace",
    help="Pre-tokenizer to use.",
)
parser.add_argument(
    "--min_freq",
    type=int,
    default=2,
    help="Minimum frequency for a word to be included in the vocabulary.",
)
parser.add_argument(
    "--output",
    type=str,
    default="wikizh_tokenizer.json",
    help="Path to save the trained tokenizer.",
)
args = parser.parse_args()

tokenizer = Tokenizer(models.BPE(unk_token="<|endoftext|>"))

if args.pre_tokenizer == "Whitespace":
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
elif args.pre_tokenizer == "ByteLevel":
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=args.vocab_size,
    special_tokens=["<|endoftext|>"],
    min_frequency=args.min_freq,
)

input_path = Path(args.input)
if input_path.is_file():
    train_files = [str(input_path)]
elif input_path.is_dir():
    train_files = [str(p) for p in iter_wiki_files(input_path)]
    if not train_files:
        raise SystemExit(f"No wiki_* files found under directory: {input_path}")
    print(f"Training BPE on {len(train_files)} wiki JSON line files.")
else:
    raise SystemExit(f"Not a file or directory: {input_path}")

tokenizer.train(train_files, trainer)

tokenizer.save(args.output)
