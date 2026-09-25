import sys

from src.cli import run_cli


def main() -> None:
    """Run the application command-line interface."""

    try:
        run_cli()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
