#!/usr/bin/env python3
"""Closed-loop streaming A/B experiment. Python 3.10+, standard library only."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.request import Request, urlopen


def fetch_json(url):
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def validate_manifest(manifest):
    required = ("runtime_version", "model", "model_revision", "tokenizer_revision",
                "gpu", "driver", "cuda", "environment_artifact", "server_args",
                "prefix_caching", "hourly_cost")
    if any(key not in manifest for key in required):
        raise ValueError("Manifest is incomplete")
    for key in required[:-3]:
        if not isinstance(manifest[key], str) or not manifest[key].strip() or "REPLACE" in manifest[key]:
            raise ValueError(f"Fill manifest field: {key}")
    for key in ("model_revision", "tokenizer_revision"):
        value = manifest[key]
        if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"{key} must be an immutable 40-character commit")
    if manifest["runtime_version"] != "0.30.0":
        raise ValueError("This adapter targets vLLM 0.30.0; create a new adapter for other versions")
    if type(manifest["prefix_caching"]) is not bool:
        raise ValueError("prefix_caching must be boolean")
    args = manifest["server_args"]
    if not isinstance(args, list) or not args or any(not isinstance(x, str) for x in args):
        raise ValueError("server_args must contain the actual argument vector")
    flags = [x for x in args if x in ("--enable-prefix-caching", "--no-enable-prefix-caching")]
    expected = "--enable-prefix-caching" if manifest["prefix_caching"] else "--no-enable-prefix-caching"
    if flags != [expected]:
        raise ValueError("server_args must explicitly match prefix_caching")
    for flag, key in (("--revision", "model_revision"), ("--tokenizer-revision", "tokenizer_revision")):
        if flag not in args or args.index(flag) + 1 >= len(args) or args[args.index(flag) + 1] != manifest[key]:
            raise ValueError(f"server_args must pin {flag}")
    cost = manifest["hourly_cost"]
    if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
        raise ValueError("hourly_cost must be the positive total hourly cost of this serving instance")


def request_one(base_url, model, item, max_tokens, timeout):
    payload = {"model": model, "prompt": item["prompt"], "temperature": 0,
               "seed": 0, "max_tokens": max_tokens, "stream": True,
               "stream_options": {"include_usage": True}}
    started = time.perf_counter()
    first = last = None
    text, usage, done, finish = "", None, False, None
    record = {"id": item["id"], "expected": item["expected"], "ok": False, "quality_pass": False}
    try:
        request = Request(base_url + "/v1/completions", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=timeout) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                if event.get("error"):
                    raise ValueError(str(event["error"]))
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    chunk = choice.get("text", "")
                    if chunk:
                        now = time.perf_counter()
                        first = now if first is None else first
                        last = now
                        text += chunk
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
        if not done or not usage or first is None or finish is None:
            raise ValueError("Incomplete stream, missing usage, finish reason or output")
        count = usage["completion_tokens"]
        if type(count) is not int or count < 1:
            raise ValueError("Invalid completion_tokens")
        record.update(ok=True, quality_pass=text.strip() == item["expected"].strip(),
                      output_tokens=count, prompt_tokens=usage["prompt_tokens"],
                      ttft_seconds=first - started,
                      tpot_seconds=(last - first) / (count - 1) if count > 1 else None,
                      finish_reason=finish)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    record.update(text=text, e2e_seconds=time.perf_counter() - started)
    return record


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def summarize(records, wall_seconds, hourly_cost, ttft_slo, e2e_slo):
    successful = [r for r in records if r["ok"]]
    qualified = [r for r in successful if r["quality_pass"] and
                 r["ttft_seconds"] <= ttft_slo and r["e2e_seconds"] <= e2e_slo]
    if not records or wall_seconds <= 0:
        raise ValueError("Need nonempty results and positive wall duration")
    report = {
        "requests": len(records), "successful": len(successful), "qualified": len(qualified),
        "error_rate": 1 - len(successful) / len(records),
        "quality_rate_all_requests": sum(r["quality_pass"] for r in records) / len(records),
        "successful_output_tokens_per_second": sum(r["output_tokens"] for r in successful) / wall_seconds,
        "qualified_requests_per_second": len(qualified) / wall_seconds,
        "measurement_cost": hourly_cost * wall_seconds / 3600,
        "cost_per_qualified_request": hourly_cost * wall_seconds / 3600 / len(qualified) if qualified else None,
    }
    for metric in ("ttft_seconds", "tpot_seconds", "e2e_seconds"):
        values = [r[metric] for r in successful if r[metric] is not None]
        report[metric] = {"n": len(values), "p50": percentile(values, .5),
                          "p95": percentile(values, .95), "p99": percentile(values, .99)}
    return report


def run(args):
    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    if min(args.repeats, args.concurrency, args.max_tokens) < 1 or args.warmup_rounds < 1:
        raise ValueError("Use positive request settings and at least one warm-up round")
    for value in (args.timeout, args.ttft_slo, args.e2e_slo):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Timeout and SLO values must be positive and finite")
    dataset = args.dataset.read_bytes()
    items = [json.loads(line) for line in dataset.decode().splitlines() if line.strip()]
    if not items or any(any(not isinstance(x.get(k), str) or not x[k] for k in ("id", "prompt", "expected")) for x in items):
        raise ValueError("Dataset rows need nonempty id, prompt, expected strings")
    if len({x["id"] for x in items}) != len(items):
        raise ValueError("Dataset IDs must be unique")
    base = args.base_url.rstrip("/")
    version = fetch_json(base + "/version")
    if version.get("version") != manifest["runtime_version"]:
        raise ValueError("Server version differs from manifest")
    models = fetch_json(base + "/v1/models")
    if manifest["model"] not in {m["id"] for m in models["data"]}:
        raise ValueError("Manifest model is not served by this endpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Refuse overwriting evidence. No GPU, driver or server flag is inferred from HTTP.
    with args.output.open("x", encoding="utf-8") as out:
        settings = {k: getattr(args, k) for k in ("repeats", "concurrency", "max_tokens", "warmup_rounds", "timeout", "ttft_slo", "e2e_slo")}
        header = {"kind": "metadata", "manifest": manifest, "settings": settings,
                  "dataset_sha256": hashlib.sha256(dataset).hexdigest(),
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "timestamp": datetime.now(timezone.utc).isoformat(), "base_url": base,
                  "server_version": version, "server_models": models,
                  "load_mode": "closed-loop", "cache_state": "full-dataset warmup after server restart"}
        out.write(json.dumps(header, ensure_ascii=False) + "\n")
        out.flush()
        call = lambda item: request_one(base, manifest["model"], item, args.max_tokens, args.timeout)
        # Retain warm-up failures, but exclude warm-up from measurement denominators.
        for index, item in enumerate(items * args.warmup_rounds):
            record = call(item)
            out.write(json.dumps({"kind": "warmup", "index": index, **record}, ensure_ascii=False) + "\n")
            out.flush()
            if not record["ok"]:
                raise ValueError("Warm-up failed; evidence retained, measurement aborted")
        records = []
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            # map preserves dataset order in the log; per-request timing starts in the worker.
            for index, record in enumerate(pool.map(call, items * args.repeats)):
                records.append(record)
                out.write(json.dumps({"kind": "request", "index": index, **record}, ensure_ascii=False) + "\n")
                out.flush()
        duration = time.perf_counter() - started
        summary = summarize(records, duration, manifest["hourly_cost"], args.ttft_slo, args.e2e_slo)
        out.write(json.dumps({"kind": "summary", "wall_seconds": duration, **summary}) + "\n")
    print(json.dumps(summary, indent=2))


def read_run(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or rows[0].get("kind") != "metadata" or rows[-1].get("kind") != "summary":
        raise ValueError("Incomplete experiment log")
    requests = [r for r in rows if r.get("kind") == "request"]
    summary = rows[-1]
    header = rows[0]
    expected = summarize(requests, summary["wall_seconds"], header["manifest"]["hourly_cost"],
                         header["settings"]["ttft_slo"], header["settings"]["e2e_slo"])
    if any(summary[k] != value for k, value in expected.items()):
        raise ValueError("Summary differs from raw request records")
    return header, summary


def compare(args):
    baseline, before = read_run(args.baseline)
    candidate, after = read_run(args.candidate)
    for key in ("settings", "dataset_sha256", "script_sha256", "cache_state", "load_mode"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Incomparable runs: {key} differs")
    for header, enabled in ((baseline, False), (candidate, True)):
        validate_manifest(header["manifest"])
        if header["manifest"]["prefix_caching"] is not enabled:
            raise ValueError("Expected cache-disabled baseline and cache-enabled candidate")
    def controlled(manifest):
        manifest = dict(manifest)
        manifest.pop("prefix_caching")
        manifest["server_args"] = [x for x in manifest["server_args"] if x not in
                                   ("--enable-prefix-caching", "--no-enable-prefix-caching")]
        return manifest
    if controlled(baseline["manifest"]) != controlled(candidate["manifest"]):
        raise ValueError("Only prefix caching may change in this experiment")
    if not 0 <= args.min_quality <= 1 or not 0 <= args.max_quality_drop <= 1:
        raise ValueError("Quality thresholds must be in [0, 1]")
    passed = (after["error_rate"] == 0 and
              after["quality_rate_all_requests"] >= args.min_quality and
              after["quality_rate_all_requests"] >= before["quality_rate_all_requests"] - args.max_quality_drop and
              after["qualified_requests_per_second"] > before["qualified_requests_per_second"] and
              after["cost_per_qualified_request"] is not None and
              before["cost_per_qualified_request"] is not None and
              after["cost_per_qualified_request"] < before["cost_per_qualified_request"])
    print(json.dumps({"baseline": before, "candidate": after, "gate_passed": passed,
                      "min_quality": args.min_quality, "max_quality_drop": args.max_quality_drop,
                      "decision": "repeat alternating A/B runs before rollout" if passed else "do not promote; retain baseline"}, indent=2))
    return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--manifest", type=Path, required=True)
    runner.add_argument("--dataset", type=Path, default=Path(__file__).with_name("workload.jsonl"))
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--base-url", default="http://127.0.0.1:8000")
    runner.add_argument("--repeats", type=int, default=10)
    runner.add_argument("--concurrency", type=int, default=1)
    runner.add_argument("--warmup-rounds", type=int, default=1)
    runner.add_argument("--max-tokens", type=int, default=16)
    runner.add_argument("--timeout", type=float, default=60)
    runner.add_argument("--ttft-slo", type=float, required=True)
    runner.add_argument("--e2e-slo", type=float, required=True)
    comparison = commands.add_parser("compare")
    comparison.add_argument("--baseline", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    comparison.add_argument("--min-quality", type=float, default=1.0)
    comparison.add_argument("--max-quality-drop", type=float, default=0.0)
    args = parser.parse_args()
    try:
        return compare(args) if args.command == "compare" else run(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Experiment failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
