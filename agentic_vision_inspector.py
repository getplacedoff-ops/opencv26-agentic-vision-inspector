#!/usr/bin/env python3
"""
Agentic Vision Inspector (OpenCV 5 + AWS Bedrock AgentCore / COOL Ready)
Author: Vigneshwara Kireeti Alugunulla

Implements a closed-loop Perception -> Decision -> Action computer vision agent
with:
  1. Quantitative Perceptual Quality Objective Function J(theta) combining Tenengrad
     Sobel focus energy, Shannon information entropy, and non-saturated dynamic range.
  2. Test-Time Hyperparameter Optimization over OpenCV recovery kernels.
  3. Automatic State Backtracking / Rollback if an action over-amplifies substrate noise.
  4. Hybrid Dual-Mode Reasoning: <10ms deterministic OpenCV loop + IPC-A-610 root-cause
     morphology classification for Amazon Bedrock AgentCore (MCP v1.0) escalation.
"""

import cv2
import json
import math
import time
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Any, Optional


@dataclass
class PerceptionMetrics:
    laplacian_sharpness: float
    glare_fraction: float
    mean_luminance: float
    contrast_std: float
    edge_density: float
    candidate_regions: int
    perceptual_objective_j: float = 0.0
    shannon_entropy: float = 0.0
    tenengrad_energy: float = 0.0


@dataclass
class AgentStepTrace:
    iteration: int
    perception: Dict[str, float]
    selected_tool: str
    tool_params: Dict[str, Any]
    rationale: str
    objective_before: float = 0.0
    objective_after: float = 0.0
    backtracked: bool = False


@dataclass
class DefectDiagnosticReport:
    bbox: Tuple[int, int, int, int]
    predicted_class: str
    ipc_a_610_code: str
    confidence: float
    snr_db: float
    severity: str
    root_cause_hypothesis: str
    recommended_fab_action: str


class StaticBaselineInspector:
    """Stage 2 Baseline: Fixed single-pass Gaussian blur + Otsu/Canny contour detector."""

    def inspect(self, image_bgr: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], float]:
        t0 = time.perf_counter()
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        edges = cv2.Canny(blurred, 60, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h, w = gray.shape
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if 140 <= area <= 0.25 * (w * h):
                boxes.append((int(x), int(y), int(bw), int(bh)))
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return boxes, latency_ms


class AgenticOpenCVInspector:
    """
    Stage 3 Architecture: Closed-Loop Self-Optimizing Agentic Vision Inspector.
    Uses OpenCV perception diagnostics and a quantitative perceptual objective J(theta)
    to iteratively invoke, tune, and verify targeted OpenCV recovery tools before
    executing final defect segmentation, ROI SNR gating, and IPC-A-610 root-cause classification.
    """

    def __init__(self, max_iterations: int = 4, enable_parameter_search: bool = True):
        self.max_iterations = max_iterations
        self.enable_parameter_search = enable_parameter_search

    def compute_perceptual_objective(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Evaluates the Perceptual Quality Objective Function J(theta):
          J(I) = rho_valid * H(I) * log2(1 + G_Tenengrad) * penalty_noise
        where:
          - H(I) is the 256-bin Shannon entropy in bits
          - G_Tenengrad is mean Sobel gradient magnitude
          - rho_valid penalizes pixel saturation (>242) and severe underexposure (<25)
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
        prob = hist / float(gray.size + 1e-9)
        nonzero = prob[prob > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)
        tenengrad = float(np.mean(grad_mag))

        glare_frac = float(np.mean(gray > 242))
        dark_frac = float(np.mean(gray < 25))
        rho_valid = max(0.05, 1.0 - 2.5 * glare_frac - 0.8 * dark_frac)

        # Penalize over-sharpening speckle explosion if high-frequency edge density exceeds 0.20
        edges = cv2.Canny(gray, 50, 140)
        edge_dens = float(np.mean(edges > 0))
        noise_penalty = 1.0 if edge_dens <= 0.18 else max(0.2, 1.0 - 3.5 * (edge_dens - 0.18))

        j_score = rho_valid * entropy * math.log2(1.0 + tenengrad) * noise_penalty
        return {
            "j_score": round(j_score, 4),
            "shannon_entropy": round(entropy, 4),
            "tenengrad_energy": round(tenengrad, 3),
            "rho_valid": round(rho_valid, 4),
            "edge_density": round(edge_dens, 4),
        }

    def perceive(self, gray: np.ndarray) -> PerceptionMetrics:
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        glare_mask = (gray > 242).astype(np.uint8)
        glare_frac = float(np.mean(glare_mask))
        mean_lum = float(np.mean(gray))
        contrast = float(np.std(gray))
        edges = cv2.Canny(gray, 50, 140)
        edge_dens = float(np.mean(edges > 0))
        num_labels, _ = cv2.connectedComponents((edges > 0).astype(np.uint8))
        obj = self.compute_perceptual_objective(gray)
        return PerceptionMetrics(
            laplacian_sharpness=round(lap_var, 2),
            glare_fraction=round(glare_frac, 4),
            mean_luminance=round(mean_lum, 2),
            contrast_std=round(contrast, 2),
            edge_density=round(edge_dens, 4),
            candidate_regions=int(num_labels - 1),
            perceptual_objective_j=obj["j_score"],
            shannon_entropy=obj["shannon_entropy"],
            tenengrad_energy=obj["tenengrad_energy"],
        )

    def tool_inpaint_specular_glare(self, gray: np.ndarray, threshold: int = 220, dilation_ksize: int = 19) -> np.ndarray:
        """Suppresses bright specular reflection domes and their Gaussian halos via dilated inpainting + local median replacement."""
        _, core_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_ksize, dilation_ksize))
        halo_mask = cv2.dilate(core_mask, kernel, iterations=2)
        bg_med = float(np.median(gray[halo_mask == 0])) if np.any(halo_mask == 0) else 128.0
        repaired = gray.copy()
        repaired[halo_mask > 0] = int(round(bg_med))
        return repaired

    def tool_clahe_contrast_recovery(self, gray: np.ndarray, clip_limit: float = 3.2, tile_grid: int = 8) -> np.ndarray:
        """Recovers underexposed or low-contrast surface detail using local CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
        return clahe.apply(gray)

    def optimize_clahe_parameters(self, gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Test-time hyperparameter search over CLAHE clipLimit to maximize perceptual objective J(theta)."""
        if not self.enable_parameter_search:
            out = self.tool_clahe_contrast_recovery(gray, clip_limit=3.2)
            return out, {"clip_limit": 3.2, "tile_grid": 8, "search_candidates": 1}

        candidates = [2.4, 3.2, 4.0]
        best_img = None
        best_j = -1.0
        best_clip = 3.2
        for clip in candidates:
            cand_img = self.tool_clahe_contrast_recovery(gray, clip_limit=clip)
            score = self.compute_perceptual_objective(cand_img)["j_score"]
            if score > best_j:
                best_j = score
                best_img = cand_img
                best_clip = clip
        return best_img, {"clip_limit": best_clip, "tile_grid": 8, "search_candidates": len(candidates), "optimal_j": round(best_j, 3)}

    def tool_unsharp_deconvolution(self, gray: np.ndarray, amount: float = 1.8, sigma: float = 2.0) -> np.ndarray:
        """Restores high-frequency edge gradients degraded by defocus or motion blur."""
        gaussian = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        sharpened = cv2.addWeighted(gray, 1.0 + amount, gaussian, -amount, 0)
        return sharpened

    def optimize_unsharp_parameters(self, gray: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Test-time hyperparameter search over Unsharp Masking gain to maximize focus without noise ringing."""
        if not self.enable_parameter_search:
            out = self.tool_unsharp_deconvolution(gray, amount=1.8)
            return out, {"sigma": 2.0, "amount": 1.8, "search_candidates": 1}

        candidates = [1.5, 1.8, 2.2]
        best_img = None
        best_j = -1.0
        best_amt = 1.8
        for amt in candidates:
            cand_img = self.tool_unsharp_deconvolution(gray, amount=amt, sigma=2.0)
            score = self.compute_perceptual_objective(cand_img)["j_score"]
            if score > best_j:
                best_j = score
                best_img = cand_img
                best_amt = amt
        return best_img, {"sigma": 2.0, "amount": best_amt, "search_candidates": len(candidates), "optimal_j": round(best_j, 3)}

    def verify_step_or_backtrack(self, prev_gray: np.ndarray, candidate_gray: np.ndarray) -> Tuple[np.ndarray, float, float, bool]:
        """
        Closed-loop self-verification & backtracking guard:
        Checks whether the candidate transformation corrupted the image (e.g., severe Canny noise explosion
        or >25% collapse in perceptual objective J). If degraded, rolls back to prev_gray.
        """
        obj_before = self.compute_perceptual_objective(prev_gray)
        obj_after = self.compute_perceptual_objective(candidate_gray)
        j_before = obj_before["j_score"]
        j_after = obj_after["j_score"]

        # Trigger rollback if edge density explodes past 0.28 or J drops catastrophically
        if obj_after["edge_density"] > 0.28 or (j_before > 1.0 and j_after < 0.65 * j_before):
            return prev_gray, j_before, j_before, True
        return candidate_gray, j_before, j_after, False

    def tool_morphological_defect_isolation(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Combines Black-Hat (dark micro-cracks/pits) and Top-Hat (bright solder bridges) relative to local background."""
        bg = cv2.medianBlur(gray, 31)
        diff = cv2.absdiff(gray, bg)
        diff[:8, :] = 0
        diff[-8:, :] = 0
        diff[:, :8] = 0
        diff[:, -8:] = 0

        thresh_val = max(22.0, float(np.mean(diff) + 2.6 * np.std(diff)))
        _, binary = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h, w = gray.shape
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if 120 <= area <= 0.15 * (w * h):
                roi = diff[y:y+bh, x:x+bw]
                if float(np.max(roi)) >= thresh_val * 1.15:
                    boxes.append((int(x), int(y), int(bw), int(bh)))
        return boxes

    def diagnose_defect_rois(
        self, conditioned_gray: np.ndarray, boxes: List[Tuple[int, int, int, int]]
    ) -> List[Dict[str, Any]]:
        """
        Hybrid Multimodal / Morphological Root-Cause Diagnostic Engine:
        Analyzes each isolated ROI's intensity polarity, aspect ratio, central blowhole structure,
        and gradient anisotropy to classify IPC-A-610 defect categories and synthesize fab corrective actions.
        """
        if conditioned_gray.ndim == 3:
            conditioned_gray = cv2.cvtColor(conditioned_gray, cv2.COLOR_BGR2GRAY)
        bg = cv2.medianBlur(conditioned_gray, 31)
        signed_diff = conditioned_gray.astype(np.float32) - bg.astype(np.float32)
        bg_std = max(1.0, float(np.std(bg)))
        reports: List[Dict[str, Any]] = []

        for (x, y, bw, bh) in boxes:
            roi_signed = signed_diff[y:y+bh, x:x+bw]
            peak_abs = float(np.max(np.abs(roi_signed))) if roi_signed.size > 0 else 10.0
            snr_db = round(20.0 * math.log10(max(1.1, peak_abs / bg_std)), 2)
            mean_signed = float(np.mean(roi_signed))
            cx, cy = bw // 2, bh // 2
            center_val = float(roi_signed[cy, cx]) if (bh > 2 and bw > 2) else mean_signed
            aspect = float(max(bw, bh)) / float(max(1, min(bw, bh)))

            if mean_signed > 5.0 and center_val < -10.0:
                cls_name = "pinhole_void"
                ipc = "IPC-A-610 Class 3 §5.2.2 (Solder Blowhole / Pinhole)"
                cause = "Outgassing from moisture trapped in PCB plated through-hole or wave-solder flux volatilization."
                action = "Pre-bake FR-4 panels at 125°C for 4h; inspect reflow preheat ramp rate."
            elif mean_signed > 0.0:
                if aspect >= 1.35:
                    cls_name = "short_bridge"
                    ipc = "IPC-A-610 Class 3 §10.6 (Conductive Solder Bridge)"
                    cause = "Excessive solder paste volume or stencil aperture slump bridging adjacent fine-pitch traces."
                    action = "Trigger automated stencil wipe; reduce squeegee downstop pressure by 15%."
                else:
                    cls_name = "spur_protrusion"
                    ipc = "IPC-A-610 Class 3 §10.2.3 (Metallic Spur / Spike)"
                    cause = "Under-etched copper resist contamination or nitrogen reflow solder webbing."
                    action = "Inspect photolithography developer nozzle pressure and flux activity."
            else:
                if aspect >= 1.35:
                    cls_name = "wafer_micro_scratch"
                    ipc = "SEMI M1 / IPC-A-610 §3.1 (Anisotropic Crystal/Surface Scratch)"
                    cause = "CMP (Chemical Mechanical Planarization) slurry agglomerate or wafer robotic end-effector slip."
                    action = "Quarantine wafer lot; flush CMP pad conditioner and filter slurry line."
                elif center_val < -35.0:
                    cls_name = "open_circuit"
                    ipc = "IPC-A-610 Class 3 §10.1 (Conductor Open / Trace Void)"
                    cause = "Over-etching in cupric chloride bath or photoresist delamination."
                    action = "Divert board to flying-probe continuity test; recalibrate etch conveyor speed."
                else:
                    cls_name = "mousebite_etch"
                    ipc = "IPC-A-610 Class 3 §10.2.1 (Marginal Copper Mousebite Crater)"
                    cause = "Localized etchant turbulence eroding trace width beyond 20% IPC Class 3 tolerance."
                    action = "Log trace width reduction in SPC telemetry; schedule phototool inspection."

            reports.append(asdict(DefectDiagnosticReport(
                bbox=(x, y, bw, bh),
                predicted_class=cls_name,
                ipc_a_610_code=ipc,
                confidence=round(min(0.99, 0.78 + 0.008 * snr_db), 3),
                snr_db=snr_db,
                severity="CRITICAL_REJECT" if cls_name in ("open_circuit", "short_bridge", "wafer_micro_scratch") else "MAJOR_REWORK",
                root_cause_hypothesis=cause,
                recommended_fab_action=action,
            )))
        return reports

    def inspect(self, image_bgr: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[AgentStepTrace], float]:
        t0 = time.perf_counter()
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        traces: List[AgentStepTrace] = []
        used_tools = set()

        for step in range(1, self.max_iterations + 1):
            metrics = self.perceive(gray)
            if metrics.glare_fraction > 0.004 and "inpaint_specular_glare" not in used_tools:
                used_tools.add("inpaint_specular_glare")
                candidate = self.tool_inpaint_specular_glare(gray)
                gray, j_before, j_after, rolled_back = self.verify_step_or_backtrack(gray, candidate)
                traces.append(AgentStepTrace(
                    iteration=step,
                    perception=asdict(metrics),
                    selected_tool="tool_inpaint_specular_glare",
                    tool_params={"threshold": 220, "dilation_kernel_size": 19},
                    rationale=f"High specular glare detected ({metrics.glare_fraction*100:.2f}% saturated pixels); invoking dilated glare inpainting before edge analysis.",
                    objective_before=j_before,
                    objective_after=j_after,
                    backtracked=rolled_back,
                ))
            elif (metrics.contrast_std < 4.5 or metrics.mean_luminance < 85.0) and "clahe_contrast" not in used_tools:
                used_tools.add("clahe_contrast")
                candidate, opt_params = self.optimize_clahe_parameters(gray)
                gray, j_before, j_after, rolled_back = self.verify_step_or_backtrack(gray, candidate)
                traces.append(AgentStepTrace(
                    iteration=step,
                    perception=asdict(metrics),
                    selected_tool="tool_clahe_contrast_recovery",
                    tool_params=opt_params,
                    rationale=f"Low dynamic range/underexposure (luminance={metrics.mean_luminance:.1f}, std={metrics.contrast_std:.1f}); invoking test-time optimized CLAHE (clipLimit={opt_params['clip_limit']}).",
                    objective_before=j_before,
                    objective_after=j_after,
                    backtracked=rolled_back,
                ))
            elif metrics.laplacian_sharpness < 140.0 and "unsharp_deconv" not in used_tools:
                used_tools.add("unsharp_deconv")
                candidate, opt_params = self.optimize_unsharp_parameters(gray)
                gray, j_before, j_after, rolled_back = self.verify_step_or_backtrack(gray, candidate)
                traces.append(AgentStepTrace(
                    iteration=step,
                    perception=asdict(metrics),
                    selected_tool="tool_unsharp_deconvolution",
                    tool_params=opt_params,
                    rationale=f"Defocus/motion blur detected (Laplacian variance={metrics.laplacian_sharpness:.1f} < 140.0); applying test-time optimized unsharp deconvolution (amount={opt_params['amount']}).",
                    objective_before=j_before,
                    objective_after=j_after,
                    backtracked=rolled_back,
                ))
            else:
                traces.append(AgentStepTrace(
                    iteration=step,
                    perception=asdict(metrics),
                    selected_tool="tool_morphological_defect_isolation",
                    tool_params={"median_bg_kernel": 31, "snr_gate": 1.15},
                    rationale=f"Image conditioning converged (J={metrics.perceptual_objective_j:.2f}); executing background-subtracted morphological isolation and ROI SNR verification.",
                    objective_before=metrics.perceptual_objective_j,
                    objective_after=metrics.perceptual_objective_j,
                    backtracked=False,
                ))
                break

        boxes = self.tool_morphological_defect_isolation(gray)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return boxes, traces, latency_ms
