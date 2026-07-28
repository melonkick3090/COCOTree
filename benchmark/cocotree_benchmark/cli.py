from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path
from typing import Any

from . import __version__
from .adapters.pipeline_tree import convert_pipeline_run
from .evaluator import evaluate_dataset
from .io import (
    load_ground_truth_source,
    load_prediction_source,
    read_manifest,
)
from .protocol import load_protocol
from .quality import run_quality_invariants
from .report import (
    build_run_manifest,
    prepare_evaluation_output,
    write_evaluation_outputs,
    write_json,
)


DEFAULT_PROTOCOL = Path(__file__).resolve().parent / "data" / "paper_v1.json"
PINNED_VERSIONS = {
    "numpy": "1.26.4",
    "scipy": "1.16.3",
    "pycocotools": "2.0.11",
    "sentence-transformers": "5.4.1",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cocotree-benchmark",
        description="Validate and evaluate COCOTree mask-instance tree predictions.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check benchmark dependencies.")
    doctor.add_argument(
        "--paper",
        action="store_true",
        help="Also require the sentence-transformer paper label backend.",
    )

    validate = subparsers.add_parser(
        "validate",
        help="Validate a prediction source against a frozen manifest.",
    )
    validate.add_argument("--predictions", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--output", type=Path, default=None)
    validate.add_argument(
        "--allow-extra",
        action="store_true",
        help="Report, rather than reject, image IDs outside the manifest.",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Compute HPQ, OTQ, and their components.",
    )
    evaluate.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help=(
            "COCOTree release root containing annotations/instance_nodes.jsonl, "
            "or a canonical ground-truth prediction directory/JSONL."
        ),
    )
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--label-device", default="cpu")
    evaluate.add_argument("--no-matches", action="store_true")
    evaluate.add_argument("--allow-extra", action="store_true")
    evaluate.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace only known evaluator files in an existing output directory. "
            "Unknown files are never removed."
        ),
    )

    quality = subparsers.add_parser(
        "quality-test",
        help="Run deterministic controlled metric sanity checks.",
    )
    quality.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    quality.add_argument("--output", type=Path, default=None)

    convert = subparsers.add_parser(
        "convert-pipeline",
        help="Convert public generator tree.json outputs to prediction_v1.",
    )
    convert.add_argument("--input-root", type=Path, required=True)
    convert.add_argument("--manifest", type=Path, required=True)
    convert.add_argument("--output-dir", type=Path, required=True)
    convert.add_argument("--overwrite", action="store_true")
    return parser


def _doctor(require_paper: bool) -> int:
    packages = ["numpy", "scipy", "pycocotools"]
    if require_paper:
        packages.extend(["sentence-transformers", "torch"])
    rows: list[dict[str, Any]] = []
    ok = True
    for package in packages:
        try:
            version = importlib.metadata.version(package)
            expected = PINNED_VERSIONS.get(package)
            matches = expected is None or version == expected
            if not matches:
                ok = False
            rows.append(
                {
                    "package": package,
                    "status": "ok" if matches else "version-mismatch",
                    "version": version,
                    "expected": expected or "any",
                }
            )
        except importlib.metadata.PackageNotFoundError:
            ok = False
            rows.append(
                {
                    "package": package,
                    "status": "missing",
                    "version": "",
                    "expected": PINNED_VERSIONS.get(package, "any"),
                }
            )
    print(json.dumps({"ok": ok, "dependencies": rows}, indent=2))
    return 0 if ok else 2


def _validate(args: argparse.Namespace) -> int:
    manifest_ids = read_manifest(args.manifest)
    _trees, report = load_prediction_source(
        args.predictions,
        manifest_ids,
        missing_policy="empty",
        extra_policy="ignore" if args.allow_extra else "error",
    )
    report["status"] = "pass"
    if args.output is not None:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    started_at = time.perf_counter()
    protocol = load_protocol(args.protocol)
    manifest_ids = read_manifest(args.manifest)
    predictions, validation_report = load_prediction_source(
        args.predictions,
        manifest_ids,
        missing_policy=str(protocol.get("missing_prediction_policy", "empty")),
        extra_policy=(
            "ignore"
            if args.allow_extra
            else str(protocol.get("extra_prediction_policy", "error"))
        ),
    )
    ground_truth = load_ground_truth_source(args.ground_truth, manifest_ids)
    summary_rows, per_image_rows, match_rows = evaluate_dataset(
        ground_truth,
        predictions,
        manifest_ids,
        protocol=protocol,
        label_device=args.label_device,
    )
    run_manifest = build_run_manifest(
        protocol=protocol,
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        ground_truth_path=args.ground_truth,
        prediction_path=args.predictions,
        output_dir=args.output_dir,
        image_count=len(manifest_ids),
        label_device=args.label_device,
        elapsed_seconds=time.perf_counter() - started_at,
    )
    validation_report["status"] = "pass"
    validation_report["ground_truth_images"] = len(ground_truth)
    # Preserve any prior valid run until loading, validation, and metric
    # computation have all succeeded.
    prepare_evaluation_output(args.output_dir, overwrite=bool(args.overwrite))
    write_evaluation_outputs(
        args.output_dir,
        summary_rows=summary_rows,
        per_image_rows=per_image_rows,
        match_rows=match_rows,
        validation_report=validation_report,
        run_manifest=run_manifest,
        write_matches=not args.no_matches,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "metrics": summary_rows}, indent=2))
    return 0


def _quality_test(args: argparse.Namespace) -> int:
    report = run_quality_invariants(load_protocol(args.protocol))
    if args.output is not None:
        write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def _convert_pipeline(args: argparse.Namespace) -> int:
    report = convert_pipeline_run(
        args.input_root,
        args.output_dir,
        read_manifest(args.manifest),
        overwrite=bool(args.overwrite),
    )
    report_path = args.output_dir / "conversion_report.json"
    write_json(report_path, report)
    print(
        json.dumps(
            {
                "requested_images": report["requested_images"],
                "converted_images": report["converted_images"],
                "missing_image_count": report["missing_image_count"],
                "warning_image_count": report["warning_image_count"],
                "output_dir": str(args.output_dir),
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return 0 if not report["missing_images"] else 2


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        status = _doctor(args.paper)
    elif args.command == "validate":
        status = _validate(args)
    elif args.command == "evaluate":
        status = _evaluate(args)
    elif args.command == "quality-test":
        status = _quality_test(args)
    elif args.command == "convert-pipeline":
        status = _convert_pipeline(args)
    else:
        raise AssertionError(args.command)
    raise SystemExit(status)


if __name__ == "__main__":
    main(sys.argv[1:])
