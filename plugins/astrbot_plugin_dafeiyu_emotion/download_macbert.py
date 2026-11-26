"""把 MacBERT 基座下载到本机，并核对情绪分类权重。

基座 hfl/chinese-macbert-base 来自 Hugging Face，只需要分词器和配置。
情绪判断用的 best.pt 是训练产物，公开页面没有这份文件。
把它从训练机器拷到同一个目录，脚本会核对 sha256。
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE_REPO = "hfl/chinese-macbert-base"
BASE_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "added_tokens.json",
)
CHECKPOINT_SHA256 = "38b9300f32048d87696146e2d72c8b3cad0f4e02597aec82abfc09bf84d06bbd"
MIRRORS = (
    "https://huggingface.co/{repo}/resolve/main/{name}",
    "https://hf-mirror.com/{repo}/resolve/main/{name}",
)


def download_base(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in BASE_FILES:
        target = dest / name
        if target.is_file() and target.stat().st_size > 0:
            print(f"已有 {target}")
            continue
        last_error = ""
        for pattern in MIRRORS:
            url = pattern.format(repo=BASE_REPO, name=name)
            try:
                print(f"下载 {url}")
                with urllib.request.urlopen(url, timeout=60) as response:
                    target.write_bytes(response.read())
                break
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if target.exists():
                    target.unlink()
        else:
            raise RuntimeError(f"下载 {name} 失败：{last_error}")
    print(f"基座文件在 {dest}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_checkpoint(path: Path) -> None:
    if not path.is_file():
        print("还没有情绪分类权重。请从训练机器复制：")
        print("  experiments/smp_text_emotion_v1/runs/matched_v1/macbert/seed42/best.pt")
        print(f"放到 {path}")
        print(f"期望 sha256 {CHECKPOINT_SHA256}")
        return
    actual = sha256(path)
    if actual != CHECKPOINT_SHA256:
        raise SystemExit(f"best.pt 校验不一致：{actual}")
    print(f"情绪分类权重校验通过 {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 MacBERT 基座到本地")
    parser.add_argument("--dest", default="models/chinese_macbert_base")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()
    dest = Path(args.dest)
    download_base(dest)
    checkpoint = Path(args.checkpoint) if args.checkpoint else dest / "best.pt"
    check_checkpoint(checkpoint)
    print("插件配置：")
    print(f"  macbert_base_dir = {dest.resolve()}")
    print(f"  macbert_checkpoint = {checkpoint.resolve()}")
    print("  classifier = macbert")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(exc, file=sys.stderr)
        sys.exit(1)
