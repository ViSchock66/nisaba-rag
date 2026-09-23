"""Allow ``python -m nisaba_rag`` as a shortcut for the CLI."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
