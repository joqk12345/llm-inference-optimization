import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_reference_algorithms import module

experiment = module("experiment", "code/chapter10/experiment.py")
diagnostic = module("diagnostic", "code/chapter03/bottleneck_diagnostic.py")


def stream(text="red cedar", done=True, usage=True):
    events = [{"choices": [{"text": text, "finish_reason": None}]},
              {"choices": [{"text": "", "finish_reason": "stop"}]}]
    if usage:
        events.append({"choices": [], "usage": {"prompt_tokens": 128, "completion_tokens": 2}})
    body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
    if done:
        body += "data: [DONE]\n\n"
    return io.BytesIO(body.encode())


def manifest(enabled=False):
    revision = "a" * 40
    flag = "--enable-prefix-caching" if enabled else "--no-enable-prefix-caching"
    return {"runtime_version": "0.30.0", "model": "test", "model_revision": revision,
            "tokenizer_revision": revision, "gpu": "test GPU", "driver": "test driver",
            "cuda": "test CUDA", "environment_artifact": "test digest", "hourly_cost": 1,
            "prefix_caching": enabled,
            "server_args": ["vllm", "serve", "test", "--revision", revision,
                            "--tokenizer-revision", revision, flag]}


class StreamingTests(unittest.TestCase):
    item = {"id": "1", "prompt": "lookup", "expected": "red cedar"}

    def test_stream_records_usage_and_quality(self):
        with patch.object(experiment, "urlopen", return_value=stream()):
            result = experiment.request_one("http://test", "test", self.item, 16, 5)
        self.assertTrue(result["ok"])
        self.assertTrue(result["quality_pass"])
        self.assertEqual(result["output_tokens"], 2)

    def test_incomplete_stream_is_a_failure(self):
        for kwargs in ({"done": False}, {"usage": False}):
            with self.subTest(kwargs=kwargs), patch.object(experiment, "urlopen", return_value=stream(**kwargs)):
                result = experiment.request_one("http://test", "test", self.item, 16, 5)
                self.assertFalse(result["ok"])
                self.assertFalse(result["quality_pass"])

    def test_transport_failure_is_retained(self):
        with patch.object(experiment, "urlopen", side_effect=TimeoutError("test")):
            result = experiment.request_one("http://test", "test", self.item, 16, 5)
        self.assertFalse(result["ok"])
        self.assertIn("TimeoutError", result["error"])

    def test_wrong_answer_and_failure_stay_in_quality_denominator(self):
        success = {"ok": True, "quality_pass": True, "ttft_seconds": .1,
                   "e2e_seconds": .2, "tpot_seconds": None, "output_tokens": 1}
        wrong = {**success, "quality_pass": False}
        failed = {"ok": False, "quality_pass": False}
        result = experiment.summarize([success, wrong, failed], 3, 3600, 1, 1)
        self.assertEqual(result["quality_rate_all_requests"], 1 / 3)
        self.assertEqual(result["qualified_requests_per_second"], 1 / 3)
        self.assertEqual(result["cost_per_qualified_request"], 3)
        self.assertIsNone(result["tpot_seconds"]["p95"])

    def test_run_writes_complete_log_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "environment.json").write_text(json.dumps(manifest()))
            (root / "data.jsonl").write_text(json.dumps(self.item) + "\n")
            args = SimpleNamespace(manifest=root / "environment.json", dataset=root / "data.jsonl",
                                   output=root / "result.jsonl", base_url="http://test", repeats=2,
                                   concurrency=1, max_tokens=16, warmup_rounds=1, timeout=5,
                                   ttft_slo=2, e2e_slo=5)
            def endpoint(url):
                return {"version": "0.30.0"} if url.endswith("/version") else {"data": [{"id": "test"}]}
            with patch.object(experiment, "fetch_json", side_effect=endpoint), \
                 patch.object(experiment, "urlopen", side_effect=lambda *a, **k: stream()), \
                 contextlib.redirect_stdout(io.StringIO()):
                experiment.run(args)
                header, summary = experiment.read_run(args.output)
                self.assertEqual(summary["requests"], 2)
                self.assertEqual(summary["qualified"], 2)
                self.assertEqual(header["cache_state"], "full-dataset warmup after server restart")
                with self.assertRaises(FileExistsError):
                    experiment.run(args)
            with args.output.open("a") as out:
                out.write(json.dumps({"kind": "request"}) + "\n")
            with self.assertRaises(ValueError):
                experiment.read_run(args.output)

    def test_manifest_rejects_ambiguous_flags(self):
        data = manifest()
        data["server_args"].append("--enable-prefix-caching")
        with self.assertRaises(ValueError):
            experiment.validate_manifest(data)

    def test_comparison_rejects_extra_variable_and_failed_quality(self):
        header = {"manifest": manifest(), "settings": {}, "dataset_sha256": "same",
                  "script_sha256": "same", "cache_state": "warm", "load_mode": "closed-loop"}
        after_header = {**header, "manifest": manifest(True)}
        before = {"error_rate": 0, "quality_rate_all_requests": 1,
                  "qualified_requests_per_second": 1, "cost_per_qualified_request": 1}
        after = {**before, "qualified_requests_per_second": 2, "cost_per_qualified_request": .5}
        args = SimpleNamespace(baseline=None, candidate=None, min_quality=1, max_quality_drop=0)
        with patch.object(experiment, "read_run", side_effect=[(header, before), (after_header, after)]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiment.compare(args), 0)
        after["quality_rate_all_requests"] = .5
        with patch.object(experiment, "read_run", side_effect=[(header, before), (after_header, after)]), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(experiment.compare(args), 2)
        after_header["manifest"]["server_args"] += ["--max-model-len", "4096"]
        with patch.object(experiment, "read_run", side_effect=[(header, before), (after_header, after)]):
            with self.assertRaises(ValueError):
                experiment.compare(args)


class SamplingTests(unittest.TestCase):
    def test_sampling_occurs_during_work_and_phases_are_separate(self):
        active = threading.Event()
        def work():
            active.set()
            time.sleep(.04)
            active.clear()
        observed = []
        def sample():
            observed.append(active.is_set())
            return 90, 20
        result = diagnostic.measure_phase(work, sample, .005)
        self.assertTrue(observed)
        self.assertTrue(any(observed))
        self.assertEqual(result["gpu_busy_percent"], 90)
        self.assertFalse(result["sufficient_observation"])
        other = diagnostic.measure_phase(work, lambda: (10, 80), .005)
        self.assertEqual(other["gpu_busy_percent"], 10)
        self.assertEqual(other["diagnosis"], "UNDETERMINED")

    def test_missing_and_failed_counters_do_not_invent_diagnosis(self):
        result = diagnostic.measure_phase(lambda: None)
        self.assertIsNone(result["gpu_busy_percent"])
        self.assertFalse(result["sufficient_observation"])
        def unavailable():
            raise RuntimeError("counter unavailable")
        result = diagnostic.measure_phase(lambda: time.sleep(.02), unavailable, .001)
        self.assertIn("counter unavailable", result["error"])
        self.assertEqual(result["diagnosis"], "UNDETERMINED")


if __name__ == "__main__":
    unittest.main()
