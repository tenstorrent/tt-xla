import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from tools.triage_ci_failure import build_packet, packet_to_markdown, parse_github_actions_job_url


class TestTriageCiFailure(unittest.TestCase):
    def build_packet_from_text(self, text: str):
        return self.build_packet_from_text_with_source(text, "job-log")

    def build_packet_from_text_with_source(self, text: str, input_source: str):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "job.log"
            log_path.write_text(text, encoding="utf-8")
            args = Namespace(
                job_log=str(log_path),
                junit=None,
                run_url="https://github.com/tenstorrent/tt-xla/actions/runs/1",
                job_url="https://github.com/tenstorrent/tt-xla/actions/runs/1/job/2",
                workflow_name=None,
                job_name=None,
                attempt=None,
                conclusion=None,
                input_source=input_source,
                repo=None,
            )
            return build_packet(args)

    def test_pcc_failure_extracts_values_and_repro(self):
        packet = self.build_packet_from_text(
            "tests/torch/models/Hunyuan_1_5/test_text_encoder.py::test_text_encoder_sharded "
            "AssertionError: Evaluation result 0 failed: PCC comparison failed. "
            "Calculated: pcc=0.9899976894685006. Required: pcc=0.99. n300-llmbox"
        )

        self.assertEqual(packet.failure_class, "pcc_failure")
        self.assertEqual(packet.pcc_calculated, "0.9899976894685006")
        self.assertEqual(packet.pcc_required, "0.99")
        self.assertEqual(
            packet.test_selector,
            "tests/torch/models/Hunyuan_1_5/test_text_encoder.py::test_text_encoder_sharded",
        )
        self.assertEqual(
            packet.repro_command,
            "pytest -svv tests/torch/models/Hunyuan_1_5/test_text_encoder.py::test_text_encoder_sharded",
        )
        self.assertEqual(packet.suggested_owner_area, "model quality or numerical comparison")

    def test_missing_shared_library_is_infra_failure(self):
        packet = self.build_packet_from_text(
            "tests/integrations/vllm_plugin/generative/test_decode.py::test_decode\n"
            "Failed to import from vllm._C: ImportError('libcudart.so.12: cannot open shared object file: No such file or directory')"
        )

        self.assertEqual(packet.failure_class, "infra_failure")
        self.assertIn("libcudart.so.12", packet.evidence[0].text)
        self.assertEqual(packet.suggested_owner_area, "CI or environment infrastructure")

    def test_llvm_grid_mismatch_overrides_import_excerpt(self):
        packet = self.build_packet_from_text(
            "FAILURE REASON:\n"
            "Failed to import from vllm._C: ImportError('libcudart.so.12: cannot open shared object file: No such file or directory')\n"
            "LLVM ERROR: OpModel device worker grid does not match the registered system descriptor: "
            "device compute-with-storage grid {y=10, x=13}, system desc grid {y=10, x=11}.\n"
        )

        self.assertEqual(packet.failure_class, "compile_failure")
        self.assertEqual(packet.reason, "LLVM OpModel worker-grid mismatch matched.")
        self.assertEqual(packet.evidence[0].kind, "llvm_grid_mismatch")
        self.assertEqual(packet.suggested_owner_area, "StableHLO/TTIR/TTNN lowering or compiler integration")

    def test_issue_body_input_downgrades_confidence_and_warns(self):
        packet = self.build_packet_from_text_with_source(
            "### SUMMARY\n"
            "FAILURE REASON:\n"
            "Failed to import from vllm._C: ImportError('libcudart.so.12: cannot open shared object file: No such file or directory')",
            "issue-body",
        )
        markdown = packet_to_markdown(packet)

        self.assertEqual(packet.failure_class, "infra_failure")
        self.assertEqual(packet.confidence, "medium")
        self.assertEqual(packet.input_source, "issue_body")
        self.assertEqual(packet.evidence_completeness, "incomplete")
        self.assertFalse(packet.ready_to_post)
        self.assertEqual(packet.post_blockers, ["full_log_evidence_missing"])
        self.assertIn("Classification is preliminary", markdown)
        self.assertIn("Do not auto-post as root cause", markdown)

    def test_parse_github_actions_job_url(self):
        repo, run_id, job_id = parse_github_actions_job_url(
            "https://github.com/tenstorrent/tt-xla/actions/runs/28066486199/job/83214425085"
        )

        self.assertEqual(repo, "tenstorrent/tt-xla")
        self.assertEqual(run_id, "28066486199")
        self.assertEqual(job_id, "83214425085")

    @patch("tools.triage_ci_failure.subprocess.run")
    def test_job_url_fetches_log_with_gh(self, run_mock):
        run_mock.return_value.stdout = "LLVM ERROR: OpModel device worker grid does not match the registered system descriptor: grid"
        args = Namespace(
            job_log=None,
            junit=None,
            run_url="https://github.com/tenstorrent/tt-xla/actions/runs/28066486199",
            job_url="https://github.com/tenstorrent/tt-xla/actions/runs/28066486199/job/83214425085",
            workflow_name=None,
            job_name=None,
            attempt=None,
            conclusion=None,
            input_source="job-log",
            repo=None,
        )

        packet = build_packet(args)

        self.assertEqual(packet.failure_class, "compile_failure")
        self.assertTrue(packet.ready_to_post)
        run_mock.assert_called_once_with(
            [
                "gh",
                "run",
                "view",
                "28066486199",
                "--repo",
                "tenstorrent/tt-xla",
                "--job",
                "83214425085",
                "--log",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_multiple_selectors_without_junit_blocks_posting(self):
        packet = self.build_packet_from_text(
            "PCC comparison failed. Calculated: pcc=0.98. Required: pcc=0.99\n"
            "tests/torch/models/model_a/test_a.py::test_a\n"
            "tests/torch/models/model_b/test_b.py::test_b\n"
        )
        markdown = packet_to_markdown(packet)

        self.assertEqual(packet.failure_class, "pcc_failure")
        self.assertTrue(packet.ambiguous_context)
        self.assertFalse(packet.ready_to_post)
        self.assertIn("multiple_test_selectors_without_junit", packet.post_blockers)
        self.assertIn("Verify the failure-to-test mapping", markdown)

    def test_nearest_preceding_selector_allows_multi_selector_log(self):
        packet = self.build_packet_from_text(
            "tests/torch/models/model_a/test_a.py::test_a\n"
            "setup output\n"
            "tests/torch/models/model_b/test_b.py::test_b\n"
            "PCC comparison failed. Calculated: pcc=0.98. Required: pcc=0.99\n"
        )

        self.assertEqual(packet.failure_class, "pcc_failure")
        self.assertEqual(packet.test_selector, "tests/torch/models/model_b/test_b.py::test_b")
        self.assertFalse(packet.ambiguous_context)
        self.assertTrue(packet.ready_to_post)

    def test_parametrized_selector_keeps_closing_bracket(self):
        packet = self.build_packet_from_text(
            "tests/benchmark/test_vllm_benchmarks.py::test_vllm_benchmark[falcon3-7b-base]\n"
            "TT_FATAL: Out of Memory: Not enough space to allocate 141557760 B DRAM buffer"
        )

        self.assertEqual(
            packet.test_selector,
            "tests/benchmark/test_vllm_benchmarks.py::test_vllm_benchmark[falcon3-7b-base]",
        )

    def test_parametrized_selector_keeps_slash_and_bracket(self):
        packet = self.build_packet_from_text(
            "tests/torch/models/deepseek_v4/test_deepseek_v4_tp.py::test_transformer_decode[3-128-expected_pcc3-deepseek-ai/DeepSeek-V4-Flash]\n"
            "PCC comparison failed. Calculated: pcc=0.9899976894685006. Required: pcc=0.99"
        )

        self.assertEqual(
            packet.test_selector,
            "tests/torch/models/deepseek_v4/test_deepseek_v4_tp.py::test_transformer_decode[3-128-expected_pcc3-deepseek-ai/DeepSeek-V4-Flash]",
        )

    def test_timeout_failure_class(self):
        packet = self.build_packet_from_text(
            "tests/jax/single_chip/ops/test_add.py::test_add[shape0]\n"
            "Error: The operation timed out after 3600 seconds"
        )

        self.assertEqual(packet.failure_class, "timeout")
        self.assertEqual(packet.framework, "jax")
        self.assertEqual(packet.suggested_owner_area, "test runtime, performance, or CI capacity")

    def test_artifact_missing_failure_class(self):
        packet = self.build_packet_from_text(
            "tests/torch/graphs/test_attention.py::test_attention\n"
            "No artifacts found for test-reports-*"
        )

        self.assertEqual(packet.failure_class, "artifact_missing")
        self.assertEqual(packet.suggested_owner_area, "CI artifact collection")

    def test_pcc_summary_failure_extracts_value(self):
        packet = self.build_packet_from_text(
            "`tests/torch/models/z_image/test_transformer.py::test_transformer_sharded`:\n"
            "PCC = 0.626  (required 0.99)"
        )

        self.assertEqual(packet.failure_class, "pcc_failure")
        self.assertEqual(packet.pcc_calculated, "0.626")
        self.assertEqual(packet.pcc_required, "0.99")
        self.assertEqual(packet.test_selector, "tests/torch/models/z_image/test_transformer.py::test_transformer_sharded")

    def test_memory_failure_class(self):
        packet = self.build_packet_from_text(
            "tests/torch/models/HiDream_I1/test_transformer.py::test_transformer_sharded\n"
            "TT_FATAL: Out of Memory: Not enough space to allocate 141557760 B DRAM buffer"
        )

        self.assertEqual(packet.failure_class, "runtime_failure")
        self.assertEqual(packet.suggested_owner_area, "runtime or model loader")

    def test_unknown_failure_marks_hypothesis_and_no_local_path_in_markdown(self):
        packet = self.build_packet_from_text("unexpected failure without known markers")
        markdown = packet_to_markdown(packet)

        self.assertEqual(packet.failure_class, "unknown")
        self.assertIn("Manual inspection is required", markdown)
        self.assertNotIn("job.log", markdown)


if __name__ == "__main__":
    unittest.main()
