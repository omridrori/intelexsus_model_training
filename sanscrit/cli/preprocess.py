"""CLI wrapper around sanscrit.preprocessing.chunking.run_preprocessing"""
import argparse

from sanscrit.preprocessing.chunking import run_preprocessing


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Sanskrit corpus into clean chunks")
    parser.add_argument(
        "--raw-data-path",
        type=str,
        help="Override raw data directory (defaults to sanscrit.config.RAW_DATA_DIR)",
    )
    parser.add_argument("--output", type=str, help="Override output chunks file")
    parser.add_argument("--report", type=str, help="Override report file")
    args = parser.parse_args()

    run_preprocessing(
        raw_data_path=args.raw_data_path, output_file=args.output, report_file=args.report
    )


if __name__ == "__main__":
    main()