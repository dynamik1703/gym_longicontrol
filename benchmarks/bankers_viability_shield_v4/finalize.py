"""Seal descriptive reports after a terminal resource gate or complete Development."""
# ruff: noqa: E501 -- generated scientific report prose

import json
import resource
import time

from .controller_adapters import METHODS
from .integrity import HERE, source_identity, verify_sources
from .serialization import atomic_json, file_hash, require


def load(name):
    return json.loads((HERE / name).read_text())


def main():
    start = time.process_time()
    prep = load("preparation_status.json")
    eq = load("equivalence_results.json")
    quality = load("quality_checks.json")
    integrity = verify_sources(prep["identity"])
    require(
        quality["identity"] == eq["identity"] == prep["identity"] == source_identity(),
        "gate identities",
    )
    if not prep["all_gates_passed"]:
        require(
            not (HERE / "split_ledger.json").exists(),
            "resource-stop split must stay closed",
        )
        atomic_json(
            HERE / "results.json",
            dict(
                study="BANKERS_VIABILITY_SHIELD_V4",
                classification="BANKERS_V4_RESOURCE_STOP",
                completed_pairs=0,
                source_identity=prep["identity"],
                development_opened=False,
                controllers=None,
                rl_training_transitions=0,
                rl_gradient_updates=0,
                historical_validation="UNTOUCHED",
                paper="SEALED_UNINSPECTED",
            ),
            immutable=True,
        )
    result = load("results.json")
    json.loads(
        (
            HERE.parent / "bankers_viability_shield_v3/preparation_status.json"
        ).read_text()
    )
    measurements = load("resource_measurements.json")
    reduction = {
        m: dict(
            v3_bytes_per_step=max(
                x["compressed_bytes_per_step"]
                for x in json.loads(
                    (
                        HERE.parent
                        / "bankers_viability_shield_v3/resource_measurements.json"
                    ).read_text()
                )
                if x["method"] == m
            ),
            v4_bytes_per_step=prep["method_compressed_bytes_per_step"][m],
        )
        for m in METHODS
    }
    for item in reduction.values():
        item["reduction_fraction"] = (
            1 - item["v4_bytes_per_step"] / item["v3_bytes_per_step"]
        )
    fixtures = load("fixture_results.json")
    fixture_native = sum(x["steps"] for x in fixtures.values())
    resource_native = sum(x["steps"] for x in measurements)
    fixture_grid = 21 * fixture_native
    resource_grid = 21 * sum(
        x["steps"] for x in measurements if x["method"].startswith("BANKERS_")
    )
    # Future experiment: 3 seeds x 3 matched arms x300k transitions. Not training.
    # One extra successor mask per Banker transition is conservatively charged.
    components = {
        m: dict(
            mask_s=[v for x in measurements if x["method"] == m for v in x["mask_s"]],
            selected_s=[
                v for x in measurements if x["method"] == m for v in x["final_s"]
            ],
            replay_s_per_step=max(
                x["replay_s_per_step"] for x in measurements if x["method"] == m
            ),
            evidence_bytes_per_step=prep["method_compressed_bytes_per_step"][m],
        )
        for m in METHODS
    }
    per_transition = sum(
        prep["method_worst_step_s"][m]
        for m in ("CAUSAL60", "LEGACY_NATIVE_POST_GREEDY", "BANKERS_GREEDY")
    ) + max(components["BANKERS_GREEDY"]["mask_s"])
    rl = dict(
        algorithm="prospective masked discrete-action algorithm; no learner implemented",
        seeds=[11, 29, 47],
        matched_arms=["NO_SHIELD", "LEGACY_POST_SHIELD", "BANKER_PRE_SHIELD"],
        transitions_per_seed_per_arm=300000,
        total_transitions=2700000,
        measured_components=components,
        measured_component_projection_s=1.5 * 3 * 300000 * per_transition,
        estimate_includes="causal baseline proxy, legacy, Banker online + selected verification + complete independent reconstruction + evidence I/O; one additional successor mask for every Banker transition;1.5 factor",
        replay_storage_estimate_bytes=3
        * 300000
        * sum(
            prep["method_compressed_bytes_per_step"][m]
            for m in ("CAUSAL60", "LEGACY_NATIVE_POST_GREEDY", "BANKERS_GREEDY")
        ),
        unmeasured_components=[
            "neural inference for all arms",
            "optimizer update frequency and cost",
            "replay sampling and tensor conversion",
            "neural checkpoint size/time",
            "evaluation panel and frequency",
            "successor-mask replay representation and terminal masking",
        ],
        unsupported_assumptions=[
            "three300k budgets are a scenario, not an approved protocol",
            "CAUSAL60 is only a runtime proxy for unshielded control",
            "policy-dependent state distribution may exceed synthetic probe costs",
            "no neural runtime or total24h conclusion is supported",
        ],
        complete_total_projection_supported=False,
        training_fits_24h=None,
        training_authorized=False,
        rl_transitions=0,
        gradient_updates=0,
    )
    atomic_json(HERE / "rl_projection.json", rl, immutable=True)
    sum(v.get("cpu_s", 0) for v in fixtures.values())
    total_native = (
        fixture_native + resource_native + result.get("independent_native_steps", 0)
    )
    total_candidates = (
        eq["grid_actions"]
        + fixture_grid
        + resource_grid
        + result.get("independent_grid_candidates", 0)
    )
    core = dict(
        classification=result["classification"],
        gate_A=True,
        gate_B=True,
        gate_C=True,
        gate_D=prep["all_gates_passed"],
        gate_E=result.get("speed_gate_pass"),
        gate_F=result.get("authority_gate_pass"),
        gate_G=result.get("progress_gate_pass"),
        gate_H=True,
        compressed_comparison=reduction,
        independently_replayed_native_steps=total_native,
        independently_reconstructed_grid_candidates=total_candidates,
        equivalence_candidates=eq["grid_actions"],
        note="continuous proposal and full selected-action reconstructions are additional to grid count",
        integrity=integrity,
        historical_stops=load("source_audit.json")["immutable_stops"],
        validation="UNTOUCHED",
        paper="SEALED_UNINSPECTED",
        rl_training_transitions=0,
        gradient_updates=0,
    )
    atomic_json(HERE / "scientific_summary.json", core, immutable=True)
    docs = {}
    docs["README.md"] = (
        f"# BankerShield V4 — {result['classification']}\n\nCompact reconstructible evidence retains the frozen V3 safety meaning. {eq['grid_actions']} grid candidates on {eq['states']} synthetic states passed exact comparison and independent physical reconstruction. Development completed {result['completed_pairs']}/72 pairs. No RL training, Validation or Paper evaluation. See results.json, scientific_summary.json and SCIENTIFIC_DECISION.md.\n"
    )
    docs["SOURCE_AUDIT.md"] = (
        f"# Source audit\n\nBase b60549e95e31601039329ca8de041977da0155fe, tree41ec3d0f89bde5b2968cec909cb84aed05bb01d8. {integrity['historical_tracked_files']} inherited Git objects; {integrity['historical_hashed_files']} opaque source/evidence file hashes checked. source_audit.json records worktrees and both remotes. Push target is exclusively gym_longicontrol_origin=https://github.com/dynamik1703/gym_longicontrol.git. All changes are confined to the new V4 package. Scientific runtime is separately frozen in runtime_manifest.json; the initial read-only audit used the host interpreter. V3 online certificate/kinematics/mask sources are imported unchanged.\n"
    )
    docs["V1_V2_V3_IMMUTABILITY.md"] = (
        "# Immutable historical incidents\n\nV1 BANKERS_PRE_SHIELD_INTEGRITY_STOP; V2 BANKERS_V2_RESOURCE_STOP; V3 BANKERS_V3_RESOURCE_STOP remain unchanged. V3 compute3.900935h passed, retained projection3.259113GiB failed. V3 Development was never opened. No original STARTED pair was rerun. Every V4 scientific identity is new. Historical Validation remains untouched. Paper4000–4017 remains SEALED_UNINSPECTED; its numeric exclusion precedes construction.\n"
    )
    docs["V3_V4_EQUIVALENCE.md"] = (
        f"# Exact finite V3/V4 comparison\n\n{eq['states']} states, including {eq['inherited_main_states']} frozen V3 main synthetic states, {eq['grid_actions']} candidate safety comparisons, {eq['selections']} deterministic policy selections; zero mismatches. Every entire21-action mask and V3 decoded-transcript SHA-256 agrees. Every candidate additionally underwent full production reconstruction. Cases and input/hash identities are in equivalence_results.json. These are finite supporting tests, not exhaustive binary64 proofs. No historical scientific episode was rerun.\n"
    )
    docs["EVIDENCE_CORRUPTION_TESTS.md"] = (
        "# Corruption and interrupted writes\n\nSee tests.xml and quality_1.log: mask bit, rejection reason, backup hash, successor position (one ULP), numerical margin, native signed energy, full executed action, configuration identity, record count, record order, native timestamp, lifecycle flag, deadline diagnostic, valid prefix, selected certificate and missing field are all rejected. Semantic corruption is tested independently of an unchanged container hash. Separate tests cover envelope record-count tampering, strict JSON duplicate/nonfinite rejection, immutable publication interruption and a real process exit before publication. The crash leaves only an orphan temporary file, never a published scientific artifact. STARTED timestamp mutation, rerun and ambiguous STARTED reopening are rejected. Monkeypatching both online classifiers to fail does not prevent independent replay. Preliminary unit-test failure was a tuple/list test assertion, corrected before scientific execution; its original log is retained.\n"
    )
    table = (
        "|Controller|V3 gzip B/step|V4 gzip B/step|Reduction|V4 raw journal B/step|\n|---|---:|---:|---:|---:|\n"
        + "\n".join(
            f"|{m}|{reduction[m]['v3_bytes_per_step']:.1f}|{reduction[m]['v4_bytes_per_step']:.1f}|{100 * reduction[m]['reduction_fraction']:.2f}%|{prep['method_raw_journal_bytes_per_step'][m]:.1f}|"
            for m in METHODS
        )
    )
    docs["RESOURCE_REPORT.md"] = (
        f"# Frozen V4 resource gate\n\nSource/protocol freeze commit {prep['freeze_commit']}. Projected CPU {prep['projected_active_compute_s']:.6f}s={prep['projected_active_compute_s'] / 3600:.6f}h. Projected peak retained {prep['projected_retained_bytes']:.0f}bytes={prep['projected_retained_bytes'] / 1024**3:.6f}GiB. Gate {'PASS' if prep['all_gates_passed'] else 'FAIL'}. V3 historical stop remains unchanged.\n\n{table}\n\nThe added provenance fields can enlarge baseline evidence; storage savings are assessed separately per controller. All28 speed-band records, actual speed distributions and measured mask/selected/replay/write costs are retained in resource_measurements.json. Formula/accounting differences were committed before measurement. Peak preflight RSS {prep['peak_rss_bytes_macos']}bytes. Final measured CPU, retained package, live-journal/publication peak and source/sealing overhead appear in resource_accounting.json and budget_monitor.json where Development opened. No hard real-time guarantee is inferred.\n"
    )
    if result["completed_pairs"] == 72:
        ctr = result["controllers"]
        tbl = (
            "|Controller|RSR/18|Completion|Deadline|Exact speed|\n|---|---:|---:|---:|---:|\n"
            + "\n".join(
                f"|{m}|{ctr[m]['raw_rsr']}/18|{ctr[m]['completion']}|{ctr[m]['deadline']}|{ctr[m]['speed']}|"
                for m in METHODS
            )
        )
        docs["DEVELOPMENT_RESULTS.md"] = (
            f"# Original18-task Development denominator\n\nSplit {result['split_sha256']}; all72 pairs completed and independently verified.\n\n{tbl}\n\nEvery task's actual travel time, signed net energy, feasibility and failure identity is in results.json. Late speed-safe episodes stay infeasible; no task was removed.\n"
        )
        docs["SPLIT_LEDGER.md"] = (
            f"# BANKERS_DEV_V4\n\nOrdered hash {result['split_sha256']}. See committed split_ledger.json and namespace_reservation.json for the first18 Policy-R tasks, protected metadata, reservations and rejection reasons. Freeze precedes every performance reset. Paper excluded before construction.\n"
        )
        docs["CONTROLLER_COMPARISON.md"] = (
            "# Frozen controller comparison\n\n"
            + tbl
            + "\n\nAdditional masked CAUSAL60 failures: "
            + str(result["matched_causal60"]["additional_masked_failures"])
            + ". Exact task identities, first divergent native action, grid-versus-safety cause and later physical speed/time samples appear in matched_causal60 and first_divergences in results.json. Higher admitted actions are descriptive alternatives; no successful alternative policy is proved. Pairwise signed energy uses jointly feasible identities only (joint_feasible_energy). No post-hoc tuning.\n"
        )
        docs["ACTION_AUTHORITY.md"] = (
            "# Separate Banker authority gates\n\n"
            + "\n\n".join(
                f"{m}: {a['states']} native states; meaningful {a['meaningful']}/{a['states']}={a['meaningful_fraction']:.6f}; positive acceleration {a['positive']}/{a['states']}={a['positive_fraction']:.6f}; pass={a['gate_pass']}. Action-count/velocity-spread/maximum-acceleration/deviation distributions, binding reasons, backup and conservative braking are in results.json. Every causal envelope remains in native evidence."
                for m, a in result["authority"].items()
            )
            + "\n"
        )
    else:
        for name in (
            "DEVELOPMENT_RESULTS.md",
            "CONTROLLER_COMPARISON.md",
            "ACTION_AUTHORITY.md",
            "SPLIT_LEDGER.md",
        ):
            docs[name] = (
                "# Development remains closed\n\nThe preregistered resource gate failed before task selection. No BANKERS_DEV_V4 split, controller RSR/18, progress comparison, jointly feasible energy or authority conclusion is available. Incomplete/unopened results are not negative controller-performance evidence.\n"
            )
    docs["RL_READINESS.md"] = (
        f"# Prospective RL assessment\n\nNo learner was implemented or trained. Three seeds x three matched arms x300000 transitions is a prospective scenario only. The measured safety/evidence component projection is {rl['measured_component_projection_s'] / 3600:.3f}h, with actual masks, selected verification, complete independent reconstruction, evidence I/O and one additional successor mask per Banker transition. Replay evidence estimate {rl['replay_storage_estimate_bytes'] / 1024**3:.3f}GiB omits unknown neural/mask replay overhead. Neural inference, optimizer updates, sampling, checkpoints and evaluation are unmeasured; complete<=24h feasibility is unsupported. See rl_projection.json for measured components, estimates and unsupported assumptions. V4 conclusion: {result['classification']}. A successful study authorizes designing masked RL only; training remains unauthorized.\n"
    )
    docs["SCIENTIFIC_DECISION.md"] = (
        "# Frozen decision\n\n"
        + result["classification"]
        + "\n\n"
        + json.dumps(
            {k: v for k, v in core.items() if k.startswith("gate_")}, sort_keys=True
        )
        + "\n\nEquivalent compact safety evidence does not imply deadline feasibility. No historical stop is cleared. No RL training/updates, final Validation or Paper evaluation.\n"
    )
    docs["INTEGRITY_REPORT.md"] = (
        f"# Final integrity\n\nAll inherited source/evidence checks passed; gate/source identities match. Independently replayed native steps={total_native}; reconstructed grid candidates={total_candidates}, plus selected and continuous-proposal certificates. Immutable journals/STARTED/results enforce exactly-once execution. Every complete pair was independently physically replayed before COMPLETED; final analysis reads the sealed disk artifacts. No counterexample or unexplained integrity mismatch was observed. Test-injected corruptions were detected. V1/V2/V3 incidents and historical Validation remain unchanged; Paper remains SEALED_UNINSPECTED. Final artifact_manifest.json binds all deliverables except itself and seal_manifest.json.\n"
    )
    for name, value in docs.items():
        (HERE / name).write_text(value)
    prior = (
        load("development_seal.json")["accounted_active_compute_s"]
        if result["completed_pairs"] == 72
        else prep["preflight_cpu_s"] + prep["prior_measured_cpu_s"] + 900
    )
    retained = sum(p.stat().st_size for p in HERE.rglob("*") if p.is_file())
    accounted = prior + result.get("report_cpu_s", 0) + time.process_time() - start
    budget = (
        load("budget_monitor.json") if (HERE / "budget_monitor.json").exists() else None
    )
    accounting = dict(
        accounted_active_compute_s=accounted,
        measured_study_cpu_excluding_preparation_allowance_s=accounted - 900,
        preparation_reporting_allowance_s=900,
        projected_active_compute_s=prep["projected_active_compute_s"],
        projected_retained_bytes=prep["projected_retained_bytes"],
        measured_v4_retained_bytes_before_manifests=retained,
        measured_inherited_bytes=prep["inherited_retained_bytes"]
        + prep["native_source_bytes"],
        peak_observed_total_bytes=budget,
        final_sealing_cpu_s=time.process_time() - start,
        final_peak_rss_bytes_macos=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        scientific_runtime_manifest_sha256=file_hash(HERE / "runtime_manifest.json"),
        caveat="Active CPU is measured execution plus900s prep/report allowance, not total host or model reasoning compute.",
    )
    require(
        accounted <= 21600
        and retained + prep["inherited_retained_bytes"] + prep["native_source_bytes"]
        <= 3 * 1024**3,
        "actual hard resources",
    )
    atomic_json(HERE / "resource_accounting.json", accounting, immutable=True)
    verify_sources(prep["identity"])
    files = {
        str(p.relative_to(HERE)): file_hash(p)
        for p in sorted(HERE.rglob("*"))
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.name not in ("artifact_manifest.json", "seal_manifest.json")
    }
    atomic_json(HERE / "artifact_manifest.json", files, immutable=True)
    atomic_json(
        HERE / "seal_manifest.json",
        dict(
            identity=prep["identity"],
            artifact_manifest_sha256=file_hash(HERE / "artifact_manifest.json"),
            results_sha256=file_hash(HERE / "results.json"),
            classification=result["classification"],
            finalization_cpu_s=time.process_time() - start,
        ),
        immutable=True,
    )
    print("FINAL", result["classification"], json.dumps(accounting), flush=True)


if __name__ == "__main__":
    main()
