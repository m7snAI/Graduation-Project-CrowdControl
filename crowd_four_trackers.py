"""Add-only four-tracker comparison for the supplied CrowdControl notebook.

Run AFTER the original CELL 12, in the SAME kernel. No YOLO inference or
training is performed. No original notebook variables, CSVs, or zone outputs
are overwritten. BoT-SORT is the explicitly labelled no-ReID/no-GMC variant.

Sources:
  https://github.com/noahcao/OC_SORT
  https://docs.ultralytics.com/reference/trackers/bot_sort/
  Original notebook CELL 9 (writer and TrackEval helpers are cloned unchanged).
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import importlib.util
import inspect
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import types
import uuid

import numpy as np
import pandas as pd

SCHEMA = "crowd-four-trackers-addonly-v1"
LABEL_OC = "OC-SORT (tuned)"
LABEL_BOT = "BoT-SORT (tuned, no ReID/GMC)"
METRICS = ["HOTA", "IDF1", "MOTA", "IDSW", "FP", "FN", "DetA", "AssA", "MOTP", "MT", "ML"]


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _detections_sha(frames):
    h = hashlib.sha256()
    for frame in sorted(frames):
        h.update(str(frame).encode() + b":")
        for a in frames[frame]:
            a = np.ascontiguousarray(a, dtype=np.float32)
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def _digest(x):
    return hashlib.sha256(json.dumps(_jsonable(x), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _code_signature(code):
    """Stable across kernel restarts: ignore filenames and memory addresses."""
    constants = [_code_signature(x) if isinstance(x, types.CodeType) else repr(x)
                 for x in code.co_consts]
    return {"bytecode": code.co_code.hex(), "names": code.co_names,
            "locals": code.co_varnames, "constants": constants}


def _save_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    tmp.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _save_csv(path, df):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    df.to_csv(tmp, index=True)
    os.replace(tmp, path)


class FourTrackerComparison:
    """Read existing notebook state, then keep the extension in its own context."""

    def __init__(self, notebook_globals):
        required = ["RUNS_DRIVE", "DET_NAME", "DET_FILE", "DETS", "EMPTY", "FPS",
                    "IMG_H", "IMG_W", "EVAL_SEQUENCE", "FINAL_CSV", "BYTE_GRID_CSV",
                    "DEEP_GRID_CSV", "split_range", "find_gt", "get_track_detector",
                    "prepare_gt", "write_tracker", "run_trackeval", "_seq_name", "Detections"]
        missing = [k for k in required if k not in notebook_globals]
        if missing:
            raise RuntimeError("Run original CELL 0, CELL 8, CELL 9 and CELL 12 in this kernel first. Missing: " + ", ".join(missing))
        self._original = notebook_globals
        self.ctx = dict(notebook_globals)  # Separate namespace; no rebinding in the notebook.
        self.detector = str(notebook_globals["DET_NAME"])
        if self.detector != str(notebook_globals["get_track_detector"]()):
            raise RuntimeError("Detector selection changed. Reload CELL 8 and CELL 9 first; do not mix caches.")
        self.detections_path = Path(notebook_globals["DET_FILE"]).resolve()
        self.baseline_path = Path(notebook_globals["FINAL_CSV"]).resolve()
        for key in ["DET_FILE", "FINAL_CSV", "BYTE_GRID_CSV", "DEEP_GRID_CSV"]:
            if not Path(notebook_globals[key]).is_file():
                raise FileNotFoundError(f"Required saved output is missing: {notebook_globals[key]}")
        expected = f"detections_{self.detector}_{notebook_globals['EVAL_SEQUENCE']}.parquet"
        if self.detections_path.name != expected:
            raise RuntimeError("DET_FILE does not match the selected detector and sequence.")
        self.baselines = pd.read_csv(self.baseline_path, index_col=0)
        for name in ["ByteTrack (tuned)", "DeepSORT (tuned)"]:
            if name not in self.baselines.index:
                raise RuntimeError(f"CELL 12 result is missing row: {name}")
        if not set(METRICS).issubset(self.baselines.columns):
            raise RuntimeError("Unexpected CELL 12 CSV schema; do not substitute results from an older report.")
        if not np.isfinite(self.baselines.loc[["ByteTrack (tuned)", "DeepSORT (tuned)"], METRICS].to_numpy(float)).all():
            raise RuntimeError("The saved baseline metrics contain NaN or infinite values.")
        self.ranges = {s: tuple(map(int, self.ctx["split_range"](s))) for s in ("val", "test")}
        if self.ranges["val"][1] >= self.ranges["test"][0]:
            raise RuntimeError("VAL and TEST overlap.")
        self.h, self.w, self.fps = int(self.ctx["IMG_H"]), int(self.ctx["IMG_W"]), float(self.ctx["FPS"])
        self.frames = self.ctx["DETS"]
        if not self.frames or min(self.frames) < 1 or max(self.frames) > self.ranges["test"][1]:
            raise RuntimeError("Unexpected detection frame range.")
        # Fail rather than silently changing detections and invalidating the comparison.
        for frame, (boxes, conf) in self.frames.items():
            boxes, conf = np.asarray(boxes), np.asarray(conf)
            if boxes.shape != (len(conf), 4) or conf.ndim != 1:
                raise ValueError(f"Bad detection shapes at frame {frame}")
            if not np.isfinite(boxes).all() or not np.isfinite(conf).all():
                raise ValueError(f"Non-finite detections at frame {frame}")
            if np.any(boxes[:, 2:] <= boxes[:, :2]):
                raise ValueError(f"Non-positive box size at frame {frame}; investigate CELL 8 first.")
        self.memory_detections_hash = _detections_sha(self.frames)
        self.base_dir = Path(self.ctx["RUNS_DRIVE"]) / "tracking_four_trackers" / self.detector
        self.base_dir.mkdir(parents=True, exist_ok=True)
        # These files are read-only inputs to this extension.
        keys = ["FINAL_CSV", "BYTE_GRID_CSV", "DEEP_GRID_CSV", "SELECTION_JSON", "TRAJ_CSV", "FPS_CSV"]
        self.protected_files = {}
        for key in keys:
            if key in self.ctx and Path(self.ctx[key]).is_file():
                p = Path(self.ctx[key]).resolve()
                self.protected_files[str(p)] = _sha(p)
        self.det_stat = (self.detections_path.stat().st_size, self.detections_path.stat().st_mtime_ns)
        self.protected_names = {k: self.ctx[k] for k in ["DETS", "DET_NAME", "DET_FILE", "TE_ROOT", "FINAL_CSV",
                                "BEST_BYTE", "BEST_DEEP", "final", "BYTE_GRID_CSV", "DEEP_GRID_CSV"] if k in self.ctx}
        self._ready = False
        self._smoke_ok = False
        print("Selected detector:", self.detector)
        print("Read-only cache:", self.detections_path)
        print("VAL:", self.ranges["val"], "| TEST:", self.ranges["test"])
        print("New outputs only:", self.base_dir)
        print("Existing CELL 12 scores will be reused; no baseline numbers are hardcoded.")
        print(self.baselines.loc[["ByteTrack (tuned)", "DeepSORT (tuned)"], ["HOTA", "IDF1", "MOTA", "IDSW"]].to_string())

    def _dependencies(self):
        """No upgrades. Only a missing FilterPy is installed, in an isolated folder."""
        import ultralytics
        from ultralytics.trackers.bot_sort import BOTSORT
        self.ultralytics_version = str(ultralytics.__version__)
        self.bot_class = BOTSORT
        try:
            import filterpy.common
            import filterpy.stats
        except ModuleNotFoundError as exc:
            if not (exc.name or "").startswith("filterpy"):
                raise
            target = self.base_dir / "_dependencies"
            target.mkdir(exist_ok=True)
            if not (target / "filterpy" / "__init__.py").is_file():
                print("Installing missing FilterPy only; NumPy, PyTorch, Ultralytics and TrackEval are not upgraded.")
                subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target),
                                "filterpy==1.4.5"], check=True)
            spec = importlib.util.spec_from_file_location("filterpy", target / "filterpy" / "__init__.py",
                                                         submodule_search_locations=[str(target / "filterpy")])
            module = importlib.util.module_from_spec(spec)
            sys.modules["filterpy"] = module
            spec.loader.exec_module(module)
            import filterpy.common
            import filterpy.stats
        repo = self.base_dir / "_sources" / "OC_SORT"
        repo.parent.mkdir(exist_ok=True)
        if not repo.exists():
            if shutil.which("git") is None:
                raise RuntimeError("Git is required to retrieve the official OC-SORT source.")
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/noahcao/OC_SORT.git", str(repo)], check=True)
        if not (repo / ".git").exists():
            raise RuntimeError(f"Incomplete OC-SORT checkout: {repo}. Resolve this folder, then rerun 12B.")
        self.oc_commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        # Do not add the repository to sys.path: it contains its own trackers/TrackEval.
        module_name = "_crowd4_official_oc_" + self.oc_commit[:12]
        package_dir = repo / "trackers" / "ocsort_tracker"
        if module_name not in sys.modules:
            package = types.ModuleType(module_name)
            package.__path__ = [str(package_dir)]
            package.__package__ = module_name
            sys.modules[module_name] = package
        oc_module = importlib.import_module(module_name + ".ocsort")
        self.oc_class = oc_module.OCSort
        self.oc_module = oc_module
        self.source_hashes = {p.name: _sha(p) for p in sorted(package_dir.glob("*.py"))}
        for key in ["det_thresh", "max_age", "min_hits", "iou_threshold", "delta_t", "inertia", "use_byte"]:
            if key not in inspect.signature(self.oc_class).parameters:
                raise RuntimeError("Unexpected official OC-SORT API. Stop; do not upgrade the working project.")

    def prepare(self):
        self.check_integrity()
        self._dependencies()
        # Freeze experiment identity, including code, source versions, data and old CSVs.
        helper_names = ["_seq_name", "prepare_gt", "write_tracker", "run_trackeval"]
        helper_hashes = {n: _digest(_code_signature(self.ctx[n].__code__)) for n in helper_names}
        te = self.ctx.get("trackeval")
        te_file = getattr(te, "__file__", None)
        te_hashes = {}
        if te_file and Path(te_file).is_file():
            te_dir = Path(te_file).parent
            te_hashes = {str(p.relative_to(te_dir)): _sha(p) for p in sorted(te_dir.rglob("*.py"))}
        self.protocol = {"schema": SCHEMA, "detector": self.detector,
                         "det_cache_sha256": _sha(self.detections_path),
                         "det_memory_sha256": self.memory_detections_hash,
                         "gt_sha256": _sha(self.ctx["find_gt"](self.ctx["EVAL_SEQUENCE"])),
                         "baseline_sha256": _sha(self.baseline_path),
                         "val": self.ranges["val"], "test": self.ranges["test"],
                         "image_hw": [self.h, self.w], "fps": self.fps,
                         "ultralytics": self.ultralytics_version, "numpy": np.__version__,
                         "python": platform.python_version(), "oc_commit": self.oc_commit,
                         "oc_sources": self.source_hashes, "trackeval_sources": te_hashes,
                         "evaluation_helpers": helper_hashes,
                         "adapter_code": _digest({n: _code_signature(getattr(type(self), n).__code__)
                                          for n in ["_run", "_new_bot", "_trial", "tune", "evaluate_test"]}),
                         "oc_byte_association": False, "bot_reid": False, "bot_gmc": "none",
                         "selection": "VAL IDF1 descending, then HOTA descending",
                         "grids": self._grids(),
                         "baseline_origin": "Existing CELL 12 CSV; not rerun by this extension"}
        self.run_id = _digest(self.protocol)[:16]
        self.output_dir = self.base_dir / self.run_id
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "trials").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        _save_json(self.output_dir / "protocol.json", self.protocol)
        _save_json(self.output_dir / "protected_input_hashes.json", self.protected_files)
        local_base = Path("/content") if Path("/content").is_dir() else Path(tempfile.gettempdir())
        self.work_dir = local_base / ("crowd4_trackeval_" + self.run_id)
        self.work_dir.mkdir(exist_ok=True)
        self.eval_ctx = dict(self.ctx)
        self.eval_ctx["TE_ROOT"] = str(self.work_dir)
        # Same CELL 9 logic, but all writes resolve to an extension-only TE_ROOT.
        for name in helper_names:
            old = self.ctx[name]
            self.eval_ctx[name] = types.FunctionType(old.__code__, self.eval_ctx, name, old.__defaults__, old.__closure__)
        self._ready = True
        self._smoke_ok = False
        print("Ultralytics preserved:", self.ultralytics_version)
        print("OC-SORT checkout (not auto-updated):", self.oc_commit)
        print("TrackEval preserved:", te_file)
        print("BoT-SORT variant: with_reid=False; gmc_method='none'")
        print("Results folder:", self.output_dir)
        self.smoke_test()

    def _grids(self):
        oc = [dict(det_thresh=d, max_age=max(1, int(self.fps)), min_hits=3,
                   iou_threshold=i, delta_t=3, asso_func="iou", inertia=v, use_byte=False)
              for d, i, v in itertools.product([0.4, 0.6], [0.3, 0.5], [0.1, 0.3])]
        bot = [dict(high=h, low=0.2, new=0.7, buffer=b, match=m)
               for h, b, m in itertools.product([0.5, 0.6], [30, 60], [0.7, 0.9])]
        return {LABEL_OC: oc, LABEL_BOT: bot}

    @contextlib.contextmanager
    def _restore_counters(self):
        items = []
        try:
            from ultralytics.trackers.basetrack import BaseTrack
            items.append((BaseTrack, "_count", BaseTrack._count))
        except ImportError:
            pass
        if hasattr(self, "oc_module"):
            cls = self.oc_module.KalmanBoxTracker
            items.append((cls, "count", cls.count))
        try:
            yield
        finally:
            for obj, key, value in items:
                setattr(obj, key, value)

    def _new_bot(self, cfg):
        args = types.SimpleNamespace(tracker_type="botsort", track_high_thresh=cfg["high"],
                 track_low_thresh=cfg["low"], new_track_thresh=cfg["new"],
                 track_buffer=int(cfg["buffer"]), match_thresh=cfg["match"], fuse_score=True,
                 frame_rate=self.fps, gmc_method="none", with_reid=False, model="auto",
                 proximity_thresh=0.5, appearance_thresh=0.25)
        params = inspect.signature(self.bot_class).parameters
        tracker = self.bot_class(args, frame_rate=self.fps) if "frame_rate" in params else self.bot_class(args)
        attributes = [n for n in ("max_frames_lost", "max_time_lost") if hasattr(tracker, n)]
        if not attributes:
            raise RuntimeError("Unknown BoT-SORT buffer API. No changes have been made to your old tracker.")
        for attr in attributes:
            setattr(tracker, attr, int(self.fps / 30.0 * cfg["buffer"]))
        return tracker

    def _run(self, label, cfg, split, frames=None, frame_range=None):
        if not self._ready:
            raise RuntimeError("Run prepare() in CELL 12B first.")
        lo, hi = frame_range or self.ranges[split]
        data = self.frames if frames is None else frames
        elapsed, rows = 0.0, []
        with self._restore_counters():
            tracker = self.oc_class(**cfg) if label == LABEL_OC else self._new_bot(cfg)
            for f in range(lo, hi + 1):
                boxes, conf = data.get(f, self.ctx["EMPTY"])
                # Defensive copies prevent in-place changes to DETS by any adapter.
                if label == LABEL_OC:
                    inputs = np.column_stack((boxes, conf)).astype(np.float32, copy=True).reshape(-1, 5)
                    start = time.perf_counter()
                    tracks = tracker.update(inputs, (self.h, self.w), (self.h, self.w))
                    elapsed += time.perf_counter() - start
                else:
                    inputs = self.ctx["Detections"](np.array(boxes, copy=True), np.array(conf, copy=True))
                    start = time.perf_counter()
                    tracks = tracker.update(inputs, img=None)
                    elapsed += time.perf_counter() - start
                tracks = np.asarray(tracks)
                if tracks.size == 0:
                    continue
                min_cols = 5 if label == LABEL_OC else 6
                if tracks.ndim != 2 or tracks.shape[1] < min_cols or not np.isfinite(tracks).all():
                    raise RuntimeError(f"Unexpected {label} output at frame {f}: {tracks.shape}")
                ids = tracks[:, 4]
                if np.any(ids <= 0) or np.any(ids != np.floor(ids)) or len(np.unique(ids)) != len(ids):
                    raise RuntimeError(f"Invalid or duplicate track IDs at frame {f}")
                for row in tracks:
                    if row[2] <= row[0] or row[3] <= row[1]:
                        raise RuntimeError(f"Invalid output box from {label} at frame {f}")
                    # OC-SORT returns ID in column 4; it does NOT return a confidence.
                    score = 1.0 if label == LABEL_OC else float(row[5])
                    rows.append((f, int(row[4]), *map(float, row[:4]), score))
                if frames is None and (f - lo + 1) % 500 == 0:
                    print(f"  {label}: {f-lo+1}/{hi-lo+1}", flush=True)
        return rows, 1000.0 * elapsed / (hi - lo + 1)

    def smoke_test(self):
        if not self._ready:
            raise RuntimeError("Run prepare() first.")
        synthetic = {}
        for f in range(1, 9):
            if f in (1, 5):
                synthetic[f] = (np.empty((0, 4), np.float32), np.empty(0, np.float32))
            else:
                synthetic[f] = (np.array([[20+f, 20, 40+f, 80]], np.float32), np.array([0.95], np.float32))
        untouched = copy.deepcopy(synthetic)
        for label, configs in self._grids().items():
            rows, _ = self._run(label, configs[0], "val", synthetic, (1, 8))
            if not rows:
                raise RuntimeError(f"{label}: smoke test unexpectedly returned no tracks")
            # A short VAL-only test catches real data/API incompatibilities, not accuracy.
            lo = self.ranges["val"][0]
            last = min(lo + 19, self.ranges["val"][1])
            subset = {f: self.frames.get(f, self.ctx["EMPTY"]) for f in range(lo, last+1)}
            before = {f: (b.copy(), c.copy()) for f, (b, c) in subset.items()}
            actual, _ = self._run(label, configs[0], "val", subset, (lo, last))
            for f in subset:
                for x, y in zip(subset[f], before[f]):
                    np.testing.assert_array_equal(x, y)
            print(f"PASS: {label}: empty frames, synthetic motion, 20-frame VAL interface test ({len(actual)} rows)")
        for f in synthetic:
            for x, y in zip(synthetic[f], untouched[f]):
                np.testing.assert_array_equal(x, y)
        self.check_integrity()
        self._smoke_ok = True
        print("All preflight tests passed. This is not an accuracy result.")

    def _require_ready(self):
        if not self._ready or not self._smoke_ok:
            raise RuntimeError("CELL 12B must finish successfully before tuning/evaluation.")
        self.check_integrity()

    def _trial(self, label, cfg, split):
        trial_id = ("ocsort" if label == LABEL_OC else "botsort") + "_" + split + "_" + _digest(cfg)[:12]
        path = self.output_dir / "trials" / (trial_id + ".json")
        if path.is_file():
            result = json.loads(path.read_text())
            if result["run_id"] != self.run_id or result["cfg"] != cfg:
                raise RuntimeError("Trial cache does not match this experiment.")
            print("Loaded completed trial:", trial_id)
            return result
        rows, update_ms = self._run(label, cfg, split)
        self.eval_ctx["write_tracker"](trial_id, split, rows)
        log_path = self.output_dir / "logs" / (trial_id + ".txt")
        with open(log_path, "w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            scores = self.eval_ctx["run_trackeval"](split, [trial_id]).loc[trial_id]
        values = {k: float(scores[k]) for k in METRICS}
        if not all(np.isfinite(v) for v in values.values()):
            raise RuntimeError(f"Non-finite evaluation metrics; inspect {log_path}")
        for k in ["IDSW", "FP", "FN", "MT", "ML"]:
            values[k] = int(scores[k])
        result = {"run_id": self.run_id, "label": label, "split": split, "cfg": cfg,
                  "scores": values, "Update ms/frame": float(update_ms), "trial": trial_id}
        if split == "test":
            src = self.work_dir / "trackers" / split / trial_id / "data" / (self.eval_ctx["_seq_name"](split) + ".txt")
            dst = self.output_dir / "predictions_TEST"
            dst.mkdir(exist_ok=True)
            shutil.copy2(src, dst / (trial_id + ".txt"))
        _save_json(path, result)  # Mark completed only after evaluation and optional export succeed.
        print(f"  HOTA={values['HOTA']:.2f} | IDF1={values['IDF1']:.2f} | MOTA={values['MOTA']:.2f}")
        return result

    def tune(self):
        self._require_ready()
        selection_path = self.output_dir / "selected_on_VAL.json"
        if selection_path.is_file():
            self.selection = json.loads(selection_path.read_text())
            if self.selection["run_id"] != self.run_id:
                raise RuntimeError("Invalid saved VAL selection.")
            print("Loaded frozen VAL choices; no tuning repeated.")
            print(json.dumps(self.selection["configs"], indent=2))
            return self.selection
        if (self.output_dir / "TEST_STARTED.json").exists():
            raise RuntimeError("TEST has started; refusing to create a new selection in this experiment.")
        records, chosen = [], {}
        for label, configs in self._grids().items():
            trials = []
            for i, cfg in enumerate(configs, 1):
                print(f"\nVAL {label}: setting {i}/{len(configs)} | {cfg}", flush=True)
                r = self._trial(label, cfg, "val")
                trials.append(r)
                records.append({"tracker": label, "trial": r["trial"], "config": json.dumps(cfg, sort_keys=True),
                                **r["scores"], "Update ms/frame": r["Update ms/frame"]})
                _save_csv(self.output_dir / "new_trackers_VAL.csv", pd.DataFrame(records))
            best = sorted(trials, key=lambda r: (-r["scores"]["IDF1"], -r["scores"]["HOTA"], r["trial"]))[0]
            chosen[label] = best["cfg"]
        self.selection = {"run_id": self.run_id, "selection_split": "val", "configs": chosen}
        _save_json(selection_path, self.selection)
        self.check_integrity()
        print("\nVAL selection frozen. TEST has not been used by tuning.")
        print(json.dumps(chosen, indent=2))
        return self.selection

    def evaluate_test(self, confirm=False):
        self._require_ready()
        if confirm is not True:
            raise RuntimeError("Set CC4_RUN_TEST=True in CELL 12D only after CELL 12C finishes.")
        selection_path = self.output_dir / "selected_on_VAL.json"
        if not selection_path.is_file():
            raise RuntimeError("No VAL selection: run CELL 12C first.")
        selection = json.loads(selection_path.read_text())
        if selection["run_id"] != self.run_id or set(selection["configs"]) != {LABEL_OC, LABEL_BOT}:
            raise RuntimeError("Invalid frozen VAL selection.")
        final_path = self.output_dir / "tracking_comparison_TEST.csv"
        lock_path = self.output_dir / "TEST_STARTED.json"
        lock = {"run_id": self.run_id, "selection_sha256": _sha(selection_path)}
        if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
            raise RuntimeError("VAL choices were altered after TEST started. Refusing to reuse TEST.")
        if final_path.is_file():
            if not lock_path.exists():
                raise RuntimeError("TEST result without a frozen-selection record.")
            self.comparison = pd.read_csv(final_path, index_col=0)
            print("Loaded completed comparison; TEST not rerun.")
            return self.comparison
        _save_json(lock_path, lock)
        new = [self._trial(label, selection["configs"][label], "test") for label in (LABEL_OC, LABEL_BOT)]
        # Only accuracy metrics are merged: old pipeline FPS and new update-only FPS are not mixed.
        table = self.baselines.loc[["ByteTrack (tuned)", "DeepSORT (tuned)"], METRICS].copy()
        table["Source"] = "Existing CELL 12 TEST output"
        for r in new:
            table.loc[r["label"], METRICS] = [r["scores"][k] for k in METRICS]
            table.loc[r["label"], "Source"] = "New TEST evaluation with frozen VAL settings"
        for key in ["IDSW", "FP", "FN", "MT", "ML"]:
            table[key] = table[key].astype(int)
        self.check_integrity()
        self.comparison = table
        timing = pd.DataFrame([{"Tracker": r["label"], "Update ms/frame": r["Update ms/frame"],
                                 "Update-only FPS": 1000.0 / max(r["Update ms/frame"], 1e-12),
                                 "Scope": "update() only; excludes input packing, YOLO, ReID, I/O and rendering"} for r in new]).set_index("Tracker")
        _save_csv(self.output_dir / "new_trackers_timing_ONLY.csv", timing)
        _save_csv(final_path, table.round(3))
        print("\nFour-tracker TEST comparison:")
        print(table[["HOTA", "IDF1", "MOTA", "IDSW", "FP", "FN"]].round(2).to_string())
        print("Saved:", final_path)
        print("Original CELL 12 CSV and downstream ByteTrack outputs were not overwritten.")
        return table

    def show_results(self):
        self._require_ready()
        path = self.output_dir / "tracking_comparison_TEST.csv"
        if not path.is_file():
            raise RuntimeError("Run CELL 12D first. No TEST table exists yet.")
        table = pd.read_csv(path, index_col=0)
        from IPython.display import display
        display(table[["HOTA", "IDF1", "MOTA", "IDSW", "FP", "FN"]].round(2))
        import matplotlib.pyplot as plt
        # Separate plots; use the user's default style and color cycle.
        for metric in ["HOTA", "IDF1", "MOTA", "IDSW"]:
            fig, ax = plt.subplots(figsize=(10, 5))
            values = table[metric].to_numpy(float)
            bars = ax.bar(range(len(table)), values)
            ax.set_xticks(range(len(table)))
            ax.set_xticklabels(["ByteTrack", "DeepSORT", "OC-SORT", "BoT-SORT\n(no ReID/GMC)"])
            ax.set_ylabel(metric + (" (%)" if metric != "IDSW" else " (count)"))
            ax.set_title(f"{metric} on TEST | {self.detector} | same cached detections")
            ax.bar_label(bars, labels=[f"{v:.2f}" if metric != "IDSW" else str(int(v)) for v in values], padding=3)
            ax.margins(y=0.15)
            fig.tight_layout()
            fig.savefig(self.output_dir / ("comparison_" + metric + ".png"), dpi=180)
            plt.show()
            plt.close(fig)
        self.check_integrity()
        print("Results folder:", self.output_dir)
        print("Speed file is separate. It must not be compared with legacy end-to-end FPS.")
        return table

    def check_integrity(self):
        for name, old in self.protected_names.items():
            if self._original.get(name) is not old:
                raise RuntimeError(f"Original variable {name} was rebound. Reload CELL 12A before continuing.")
        for path, expected in self.protected_files.items():
            if not Path(path).is_file() or _sha(path) != expected:
                raise RuntimeError(f"A protected input changed during this experiment: {path}")
        if _detections_sha(self.frames) != self.memory_detections_hash:
            raise RuntimeError("The in-memory detection arrays changed. Stop and reload the experiment.")
        p = self.detections_path
        if (p.stat().st_size, p.stat().st_mtime_ns) != self.det_stat:
            raise RuntimeError("Detection cache changed during the experiment. Do not mix results.")
        return True
