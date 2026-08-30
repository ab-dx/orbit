"""orbit command-line entry point."""

from __future__ import annotations

import sys


def main() -> int:
    ver = sys.modules["orbit"].__version__
    print(f"orbit {ver}")
    print("core simulator available:", end=" ")
    try:
        from orbit import core  
    except ImportError as exc:  
        print(f"NOT built ({exc})")
        return 1
    print("built OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
