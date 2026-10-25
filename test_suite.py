#!/usr/bin/env python3
"""
Automated Unit & Integration Test Suite for OpenCV AI Competition 2026 Agentic Inspector.
Verifies:
1. Specular glare detection & dilated halo suppression (`tool_inpaint_specular_glare`)
2. Underexposed low-contrast recovery (`tool_clahe_contrast_recovery`)
3. Defocus / motion blur recovery (`tool_unsharp_deconvolution`)
4. Compound 4-step tool chaining (`glare -> clahe -> unsharp -> morphological_isolation`)
5. IPC-A-610 / DeepPCB 6-class industrial defect taxonomy generation & recovery
6. Model Context Protocol (MCP) JSON-RPC 2.0 `tools/list` and `tools/call` compliance
7. Amazon Bedrock AgentCore `lambda_handler` integration
8. AWS Graviton4 COOL hardware telemetry schema
"""

import unittest
import json
import numpy as np
from agentic_vision_inspector import AgenticOpenCVInspector, StaticBaselineInspector
from evaluate import generate_synthetic_wafer_scene, evaluate_predictions, DEFECT_CLASSES
from aws_bedrock_mcp_server import handle_jsonrpc_request, lambda_handler, get_cool_graviton_telemetry

class TestAgenticVisionInspector(unittest.TestCase):
    def setUp(self):
        self.agent = AgenticOpenCVInspector(max_iterations=4)
        self.baseline = StaticBaselineInspector()

    def test_01_glare_triggers_inpainting_tool(self):
        img, gt = generate_synthetic_wafer_scene(seed=1000, mode="glare")
        boxes, traces, _ = self.agent.inspect(img)
        tools = [t.selected_tool for t in traces]
        self.assertIn("tool_inpaint_specular_glare", tools)
        tp, fp, fn, _ = evaluate_predictions(boxes, gt)
        self.assertGreaterEqual(tp, len(gt) - 1)

    def test_02_low_contrast_triggers_clahe_tool(self):
        img, gt = generate_synthetic_wafer_scene(seed=1001, mode="underexposed_low_contrast")
        boxes, traces, _ = self.agent.inspect(img)
        tools = [t.selected_tool for t in traces]
        self.assertIn("tool_clahe_contrast_recovery", tools)
        tp, fp, fn, _ = evaluate_predictions(boxes, gt)
        self.assertEqual(fn, 0)
        self.assertEqual(fp, 0)

    def test_03_defocus_blur_triggers_unsharp_deconvolution(self):
        img, gt = generate_synthetic_wafer_scene(seed=1002, mode="defocus_blur")
        boxes, traces, _ = self.agent.inspect(img)
        tools = [t.selected_tool for t in traces]
        self.assertIn("tool_unsharp_deconvolution", tools)
        tp, fp, fn, _ = evaluate_predictions(boxes, gt)
        self.assertEqual(fn, 0)

    def test_04_compound_degradation_chains_all_four_tools(self):
        img, gt = generate_synthetic_wafer_scene(seed=1003, mode="compound_glare_and_blur")
        boxes, traces, _ = self.agent.inspect(img)
        tools = [t.selected_tool for t in traces]
        self.assertEqual(
            tools,
            [
                "tool_inpaint_specular_glare",
                "tool_clahe_contrast_recovery",
                "tool_unsharp_deconvolution",
                "tool_morphological_defect_isolation",
            ],
        )

    def test_05_six_class_deeppcb_taxonomy_generation(self):
        img, gt_boxes, gt_classes = generate_synthetic_wafer_scene(seed=1005, mode="underexposed_low_contrast", return_classes=True)
        self.assertEqual(len(gt_boxes), len(gt_classes))
        for c in gt_classes:
            self.assertIn(c, DEFECT_CLASSES)

    def test_06_mcp_jsonrpc_tools_list_and_call(self):
        list_res = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 10, "method": "tools/list"})
        self.assertEqual(len(list_res["result"]["tools"]), 5)
        call_res = handle_jsonrpc_request({
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {"name": "opencv_perceive_frame", "arguments": {"seed": 1003, "degradation_mode": "compound_glare_and_blur"}},
        })
        self.assertGreaterEqual(call_res["result"]["detected_defects"], 2)

    def test_07_bedrock_agentcore_lambda_handler(self):
        event = {
            "actionGroup": "OpenCVAgenticVisionTools",
            "function": "opencv_perceive_frame",
            "parameters": [{"name": "seed", "value": "1001"}, {"name": "degradation_mode", "value": "underexposed_low_contrast"}],
        }
        res = lambda_handler(event)
        body = json.loads(res["response"]["functionResponse"]["responseBody"]["TEXT"]["body"])
        self.assertEqual(body["degradation_mode"], "underexposed_low_contrast")

    def test_08_cool_graviton_telemetry_schema(self):
        tel = get_cool_graviton_telemetry()
        self.assertEqual(tel["cloud_economics"]["cost_reduction_percent"], 65.9)
        self.assertEqual(len(tel["kernel_benchmarks"]), 6)
        self.assertIn("live_runtime_introspection", tel)

    def test_09_perceptual_objective_and_test_time_optimization(self):
        img, _ = generate_synthetic_wafer_scene(seed=1003, mode="compound_glare_and_blur")
        gray = np.mean(img, axis=2).astype(np.uint8)
        m_before = self.agent.perceive(gray)
        self.assertGreater(m_before.perceptual_objective_j, 0.0)
        out_clahe, opt_c = self.agent.optimize_clahe_parameters(gray)
        self.assertIn(opt_c["clip_limit"], (2.4, 3.2, 4.0))
        out_unsharp, opt_u = self.agent.optimize_unsharp_parameters(out_clahe)
        self.assertGreater(opt_u["sigma"], 0.0)
        self.assertGreater(opt_u["amount"], 0.0)
        accepted_img, j_b, j_a, did_bt = self.agent.verify_step_or_backtrack(gray, out_clahe)
        self.assertIsInstance(did_bt, bool)
        self.assertEqual(accepted_img.shape, gray.shape)

    def test_10_ipc_a_610_defect_morphology_diagnostics(self):
        img, gt = generate_synthetic_wafer_scene(seed=1001, mode="underexposed_low_contrast")
        boxes, _, _ = self.agent.inspect(img)
        diags = self.agent.diagnose_defect_rois(img, boxes)
        self.assertEqual(len(diags), len(boxes))
        self.assertGreaterEqual(len(diags), 2)
        for d in diags:
            self.assertIn("predicted_class", d)
            self.assertIn("ipc_a_610_code", d)
            self.assertIn("root_cause_hypothesis", d)
            self.assertIn("recommended_fab_action", d)

    def test_11_live_opencv_simd_profiler(self):
        from cool_benchmark_profiler import introspect_opencv_runtime, run_live_simd_kernel_microbenchmark
        info = introspect_opencv_runtime()
        self.assertTrue(info["simd_optimized_enabled"])
        self.assertGreaterEqual(info["logical_cpus"], 1)
        bench = run_live_simd_kernel_microbenchmark(iterations=5)
        self.assertEqual(len(bench["live_host_kernel_timings"]), 5)

    def test_12_real_industrial_micrograph_generation_and_recovery(self):
        import tempfile
        from evaluate import generate_and_evaluate_real_industrial_micrographs
        with tempfile.TemporaryDirectory() as tmpdir:
            res, rec_path = generate_and_evaluate_real_industrial_micrographs(tmpdir, self.agent)
            self.assertEqual(len(res), 4)
            for item in res:
                self.assertGreaterEqual(item["true_positives"], item["gt_defects"] - 1)

if __name__ == "__main__":
    unittest.main(verbosity=2)
