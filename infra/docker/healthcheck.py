"""Tiny dependency-free healthcheck used by docker-compose.

The slim base image has neither curl nor procps, so healthchecks go through
the Python stdlib instead.

Usage:
    healthcheck.py http <url>          # 2xx/3xx -> healthy
    healthcheck.py process <substring> # any process whose cmdline contains it
"""

from __future__ import annotations

import os
import sys
import urllib.request


def check_http(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return 200 <= resp.status < 400
    except Exception:
        return False


def check_process(needle: str) -> bool:
    me = str(os.getpid())
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or pid == me:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmdline = fh.read().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue  # process exited while scanning
        if needle in cmdline:
            return True
    return False


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in {"http", "process"}:
        print(__doc__, file=sys.stderr)
        return 2
    ok = check_http(argv[2]) if argv[1] == "http" else check_process(argv[2])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
