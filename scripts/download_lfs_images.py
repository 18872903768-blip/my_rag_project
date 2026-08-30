"""One-off: replace Git LFS pointer image files with real content from GitHub.

The local cook corpus stores its photos as 130-byte LFS pointers.  Real bytes
live at media.githubusercontent.com/media/datawhalechina/all-in-rag/main/<path>.
Files that are already real images are skipped, so the script is idempotent.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO_MEDIA_BASE = "https://media.githubusercontent.com/media/datawhalechina/all-in-rag/main"
DATA_ROOT = Path(r"E:\Users\all_in_rag\data\C8\cook")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"
MAX_WORKERS = 8


def is_lfs_pointer(path: Path) -> bool:
    if path.stat().st_size > 1024:
        return False
    return path.read_bytes().startswith(LFS_MAGIC)


def download(root: Path, relative: Path) -> tuple[Path, str]:
    url = f"{REPO_MEDIA_BASE}/{quote(relative.as_posix())}"
    request = Request(url, headers={"User-Agent": "recipe-rag-setup"})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                payload = response.read()
            if len(payload) <= 1024:
                raise ValueError(f"下载内容仍像指针文件（{len(payload)} 字节）")
            target = root / relative
            temporary = target.with_name(target.name + ".download")
            temporary.write_bytes(payload)
            temporary.replace(target)
            return target, f"ok ({len(payload)} bytes)"
        except Exception as exc:  # noqa: BLE001 - retry then report
            last_error = exc
            time.sleep(2 * (attempt + 1))
    return root / relative, f"FAILED: {last_error}"


def main() -> int:
    pointers = [
        path
        for path in sorted(DATA_ROOT.rglob("*"))
        if path.suffix.casefold() in IMAGE_SUFFIXES and path.is_file() and is_lfs_pointer(path)
    ]
    print(f"发现 {len(pointers)} 个 LFS 指针图片，开始下载（并发={MAX_WORKERS}）")
    failures = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download, DATA_ROOT, p.relative_to(DATA_ROOT)): p for p in pointers
        }
        done = 0
        for future in as_completed(futures):
            done += 1
            target, message = future.result()
            if message.startswith("FAILED"):
                failures += 1
                print(f"[{done}/{len(pointers)}] {target.name}: {message}")
            elif done % 25 == 0:
                print(f"[{done}/{len(pointers)}] 已下载…")
    print(f"完成: 成功 {len(pointers) - failures}, 失败 {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
