"""Reproduce the September 19 review findings without changing model/config code.

Run from any directory with Python; the official-engine probe uses the configured
local RapidChiplet installation. Output is written only beneath this repository.
The ResNet reference below is a scalar Conv/Linear MAC and affine-parameter count
for torchvision's 224x224, 1000-class ResNet-50 v1.5, not a measured model run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

_bootstrap = argparse.ArgumentParser(add_help=False)
_bootstrap.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parents[1])
_bootstrap.add_argument("--output-dir", type=Path)
_options, _remaining = _bootstrap.parse_known_args()
ROOT = _options.project_root.resolve()
sys.path.insert(0, str(ROOT))

from simple_rapidchiplet.config import load_config
from simple_rapidchiplet.evaluator import evaluate_one
from simple_rapidchiplet.model import ModelSpec, Stage, load_models
from simple_rapidchiplet.preference_dse import (
    DesignState, EvaluationOracle, make_preference_profile, score_result,
)
from simple_rapidchiplet.workload import build_pipeline_workload
from simple_rapidchiplet import rapidchiplet_engine as engine

MIB_BITS = 8 * 1024**2


def resnet_reference():
    rows = []
    def row(name, cin, hin, cout, hout, params, macs):
        rows.append(dict(name=name, input_shape=[cin, hin, hin],
                         output_shape=[cout, hout, hout], params=params,
                         weight_mib=params * 4 / 1024**2, macs_g=macs / 1e9,
                         input_mib=cin * hin**2 * 4 / 1024**2,
                         output_mib=cout * hout**2 * 4 / 1024**2))
    row("stem", 3, 224, 64, 56, 3 * 64 * 49 + 2 * 64,
        112**2 * 3 * 64 * 49)
    cin, hin = 64, 56
    for stage, (planes, count) in enumerate(((64, 3), (128, 4), (256, 6), (512, 3)), 2):
        for block in range(1, count + 1):
            stride = 2 if stage > 2 and block == 1 else 1
            hout, cout = hin // stride, planes * 4
            params = cin * planes + 9 * planes**2 + planes * cout
            params += 2 * planes + 2 * planes + 2 * cout
            macs = hin**2 * cin * planes
            macs += hout**2 * (9 * planes**2 + planes * cout)
            if stride != 1 or cin != cout:
                params += cin * cout + 2 * cout
                macs += hout**2 * cin * cout
            row(f"conv{stage}_x.b{block}", cin, hin, cout, hout, params, macs)
            cin, hin = cout, hout
    row("head", 2048, 7, 1000, 1, 2048 * 1000 + 1000, 2048 * 1000)
    assert sum(r["params"] for r in rows) == 25_557_032
    return rows


def main():
    parser = argparse.ArgumentParser(parents=[_bootstrap])
    parser.add_argument("--reference-root", type=Path, default=Path(
        "C:/Users/user/Documents/GitHub/260815_cnn-e2e-constrained-ppa"))
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs/defaults.json")
    model = ModelSpec("audit_toy", "synthetic", "", 0.03, 0.001, (
        Stage("A", 0.01, 8.0, input_mb=1.0, weight_mb=0.1),
        Stage("B", 0.01, 2.0, depends_on=("A",), input_mb=8.0, weight_mb=0.1),
        Stage("C", 0.01, 0.5, depends_on=("B",), input_mb=2.0, weight_mb=0.1),
    ))
    groups = ((0, 1, 2), (1, 3, 1))
    profile = make_preference_profile("balanced", cfg=cfg)
    oc = DesignState("balanced", 3, 3, groups, model.name,
                     mapping_strategies=("output_channel", "single"))
    ic = replace(oc, mapping_strategies=("input_channel", "single"))
    result = {}
    # Force the local backend for a portable, independent cache/traffic probe.
    with patch.object(engine, "_load_rapidchiplet_module", side_effect=FileNotFoundError("audit local probe")):
        oracle = EvaluationOracle(model, cfg, profile, cfg.max_chiplets)
        cached_oc, cached_ic = oracle.evaluate(oc), oracle.evaluate(ic)
        fresh_ic = EvaluationOracle(model, cfg, profile, cfg.max_chiplets).evaluate(ic)
        result["cache"] = dict(
            requests=oracle.requests, unique_evaluations=oracle.unique_evaluations,
            same_object=cached_oc is cached_ic,
            requested_mapping=ic.mapping_strategies,
            returned_mapping=[g["strategy"] for g in cached_ic.mapping_plan["groups"]],
            cached_ic_latency_ns=cached_ic.avg_latency_ns,
            fresh_ic_latency_ns=fresh_ic.avg_latency_ns,
        )
        workload = build_pipeline_workload(model, 2, groups, mapping_strategies=oc.mapping_strategies)
        result["traffic"] = dict(
            groups=groups,
            rapid_flows_mib={f"{s}->{d}": bits / MIB_BITS
                             for (s, d), bits in workload.traffic_bits_per_inference.items()},
            mapping_edges=workload.mapping_plan["traffic_edges"],
            rapid_total_mib=workload.total_traffic_bits_per_inference / MIB_BITS,
            e2e_timings=cached_oc.e2e_boundary_timings,
            expected_boundary_mib=8.0,
            expected_0815_boundary_flows_mib={"0->2": 4.0, "1->2": 4.0},
            raw_input_mib=model.blocks[0].input_mb,
            mapping_input_mib=workload.mapping_plan["groups"][0]["input_mb"],
        )
        expensive_feasible = replace(cached_oc, avg_latency_ns=profile.latency_scale_ns * 12,
                                     total_area_mm2=0, total_power_w=0)
        infeasible = replace(expensive_feasible, architecture_feasible=False,
                             architecture_violations=("probe",))
        result["reward_ordering"] = dict(
            feasible=score_result(expensive_feasible, profile).to_dict(),
            infeasible=score_result(infeasible, profile).to_dict())

    # Capture exactly the in-memory inputs and computed intermediates consumed
    # by the real configured engine. A failure is recorded, never called success.
    try:
        rc = engine._load_rapidchiplet_module(cfg.rapidchiplet.root)
        actual_call = rc.rapidchiplet
        captures = []
        def capture(inputs, intermediates, *pos, **kw):
            output = actual_call(inputs, intermediates, *pos, **kw)
            captures.append(dict(
                chiplets=inputs["chiplets"], packaging=inputs["packaging"],
                technologies=inputs["technologies"],
                bandwidths={str(k): v for k, v in intermediates.get("link_bandwidths", {}).items()},
                link_lengths={str(k): v for k, v in intermediates.get("link_lengths", {}).items()},
            ))
            return output
        with patch.object(rc, "rapidchiplet", side_effect=capture):
            baseline = evaluate_one(model, "mesh", 3, cfg, workload=workload)
            changed_cfg = replace(cfg, network=replace(cfg.network, link_bandwidth_bits_per_cycle=1),
                                  power=replace(cfg.power, chiplet_static_w=100),
                                  rapidchiplet=replace(cfg.rapidchiplet, chiplet_power_w=123))
            changed = evaluate_one(model, "mesh", 3, changed_cfg, workload=workload)
        result["official_engine"] = dict(
            status="executed", module=str(rc.__file__), captures=captures,
            configured_bandwidth=cfg.network.link_bandwidth_bits_per_cycle,
            baseline_throughput_bits_per_cycle=baseline.aggregate_throughput_bits_per_cycle,
            changed_throughput_bits_per_cycle=changed.aggregate_throughput_bits_per_cycle,
            baseline_power_w=baseline.total_power_w,
            changed_power_w=changed.total_power_w,
            baseline_chiplet_power_w=baseline.total_chiplet_power_w,
            baseline_link_power_w=baseline.total_link_power_w,
            baseline_rapid_latency_cycles=baseline.avg_latency_cycles,
            baseline_e2e_ns=baseline.avg_latency_ns,
            changed_e2e_ns=changed.avg_latency_ns,
        )
    except (ImportError, FileNotFoundError) as exc:
        result["official_engine"] = dict(status="unavailable", error=str(exc))

    original = json.loads((args.reference_root / "研究設定.json").read_text(encoding="utf-8"))
    result["reference_0815"] = dict(root=str(args.reference_root), hardware=original["hardware"],
                                    experiment=original["experiment"],
                                    chiplet_power_w=(original["hardware"]["chiplet_static_w"]
                                        + original["hardware"]["chiplet_peak_dynamic_w"]
                                        * original["hardware"]["utilization"]
                                        + original["hardware"]["phy_w"]))
    actual_model = load_models(ROOT / "configs/models.json")["resnet50"]
    reference = resnet_reference()
    result["resnet50"] = dict(
        assumptions="N=1, 3x224x224, torchvision v1.5, FP32; Conv/Linear MACs; affine BN params included; no buffers",
        reference_source="https://docs.pytorch.org/vision/main/_modules/torchvision/models/resnet.html",
        configured_params_m=actual_model.params_m,
        configured_weight_mib=sum(b.weight_mb for b in actual_model.blocks),
        reference_params=sum(r["params"] for r in reference),
        reference_weight_mib=sum(r["weight_mib"] for r in reference),
        reference_conv_linear_macs_g=sum(r["macs_g"] for r in reference),
        blocks=[dict(reference=r, configured=dict(name=b.name, weight_mb=b.weight_mb,
                      input_mb=b.input_mb, output_mb=b.output_mb, macs_g=b.macs_g))
                for b, r in zip(actual_model.blocks, reference)],
    )
    paths = [ROOT / "configs/defaults.json", ROOT / "configs/models.json",
             *(ROOT / "simple_rapidchiplet" / n for n in (
                 "preference_dse.py", "workload.py", "mapping.py", "e2e_latency.py", "rapidchiplet_engine.py")),
             args.reference_root / "研究設定.json", args.reference_root / "rapid_integration/rapid_proxy.py"]
    result["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    out = args.output_dir or ROOT.parent / "audit_20260919"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "evidence.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"evidence": str(path), "cache": result["cache"],
                      "engine": result["official_engine"]["status"],
                      "resnet_weight_mib": result["resnet50"]["reference_weight_mib"]}, indent=2))


if __name__ == "__main__":
    main()
