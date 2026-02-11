import argparse
import os
import sys

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.loader import load_settings, resolve_paths
from ingest.file_manager import add_files_to_category
from ingest.service import run_ingest


def parse_args():
    parser = argparse.ArgumentParser(description="Add files to knowledge base and (optionally) reindex.")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml"),
        help="Path to settings.yaml",
    )
    parser.add_argument("--category", choices=["note", "paper"], help="Target category.")
    parser.add_argument("--files", nargs="*", default=[], help="Files to add.")
    parser.add_argument("--reindex", action="store_true", help="Rebuild vector index.")
    return parser.parse_args()


def main():
    args = parse_args()
    settings = load_settings(os.path.abspath(args.config))
    paths = resolve_paths(settings, os.path.abspath(args.config))

    if args.files:
        if not args.category:
            raise ValueError("--category is required when --files is provided")
        saved = add_files_to_category(paths["knowledge_base_dir"], args.category, args.files)
        print(f"Added {len(saved)} files to {args.category}")

    if args.reindex or args.files:
        count = run_ingest(settings, paths)
        print(f"Indexed vectors: {count}")


if __name__ == "__main__":
    main()
