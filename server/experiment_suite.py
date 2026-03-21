"""Experiment orchestration for scalability + robustness studies.

Runs matrix over:
- logical population N in {100,300,1000,3000}
- selected clients m in {10,20,50,100}
- split mode in {overlap, dirichlet}
- selectors in {awpsp, random, availability_only, oort}
- seeds (default 5)
- correlation noise in {0,10,20,40}

It launches main_server.py with env overrides and stores summarized outputs.
"""

import csv
import os
import subprocess
import statistics
import sys
from collections import deque
from pathlib import Path

def resolve_paths():
    script_path = Path(__file__).resolve()
    # Case A: script located at repo_root/server/experiment_suite.py
    repo_root = script_path.parents[1]
    main_server = repo_root / "server" / "main_server.py"
    if main_server.exists():
        return repo_root, main_server

    # Case B: script copied/executed directly from /app/experiment_suite.py
    repo_root = script_path.parent
    main_server = repo_root / "main_server.py"
    if main_server.exists():
        return repo_root, main_server

    raise FileNotFoundError(
        f"Could not locate main_server.py relative to {script_path}. "
        "Expected either ./server/main_server.py or ./main_server.py"
    )


ROOT, MAIN_SERVER = resolve_paths()
OUT = ROOT / "experiment_results.csv"
SUMMARY = ROOT / "experiment_summary.csv"
RUN_DIR = ROOT / "experiment_runs"


def confidence_interval(values):
    if not values:
        return 0.0, 0.0
    mean = statistics.mean(values)
    if len(values) == 1:
        return mean, 0.0
    stdev = statistics.stdev(values)
    half = 1.96 * stdev / (len(values) ** 0.5)
    return mean, half


def last_awpsp_accuracy(metrics_path):
    if not metrics_path.exists():
        return None
    with metrics_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    val = rows[-1].get("AWPSP_Accuracy")
    return float(val) if val not in (None, "") else None


def run_case(env_overrides, run_tag):
    env = os.environ.copy()
    env.update({k: str(v) for k, v in env_overrides.items()})
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = RUN_DIR / f"metrics_{run_tag}.csv"
    final_path = RUN_DIR / f"final_{run_tag}.csv"
    python_bin = os.getenv("PYTHON_BIN", sys.executable or "python3")
    cmd = [python_bin, str(MAIN_SERVER.relative_to(ROOT))]

    print(f"[suite] launching: {' '.join(cmd)}", flush=True)
    env["METRICS_LOG_PATH"] = str(metrics_path)
    env["FINAL_METRICS_PATH"] = str(final_path)
    env["REUSE_REGISTERED_CLIENTS"] = os.getenv("REUSE_REGISTERED_CLIENTS", "1")
    print(f"[suite] env: {env_overrides}", flush=True)
    print(f"[suite] metrics: {metrics_path}", flush=True)

    tail = deque(maxlen=4000)
    process = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        tail.append(line)

    process.wait()
    merged_tail = ''.join(tail)
    acc = last_awpsp_accuracy(metrics_path)
    return process.returncode, acc, merged_tail[-4000:], "", str(metrics_path), str(final_path)


def run_suite():
    print(f"[suite] ROOT={ROOT} MAIN_SERVER={MAIN_SERVER}", flush=True)
    populations = [100, 300, 1000, 3000]
    selections = [10]
    split_modes = ["overlap"]
    labels_per_client_options = [2,5,10]
    selectors = ["awpsp"]
    noises = [0]
    seeds = [0]

    rows = []
    result_fields = [
        "N", "m", "split_mode", "labels_per_client", "selector",  "noise_pct", "seed",
        "return_code", "awpsp_accuracy_last", "stdout_tail", "stderr_tail",
        "metrics_path", "final_metrics_path"
    ]
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result_fields)
        writer.writeheader()

    total = len(populations) * len(selections) * len(split_modes) * len(labels_per_client_options) * len(selectors) * len(noises) * len(seeds)
    done = 0
    for n in populations:
        for m in selections:
            for split in split_modes:
                for labels_per_client in labels_per_client_options:
                    for selector in selectors:
                        for noise in noises:
                            for seed in seeds:
                                run_tag = f"N{n}_m{m}_{split}_labels{labels_per_client}_{selector}_noise{noise}_seed{seed}"
                                code, acc, out_tail, err_tail, metrics_path, final_path = run_case(
                                    {
                                        "LOGICAL_CLIENT_COUNT": n,
                                        "LOGICAL_SELECTED_PER_ROUND": m,
                                        "LOGICAL_LABELS_PER_CLIENT": labels_per_client,
                                        "PHYSICAL_CONTAINER_LIMIT": 10,
                                        "LOGICAL_SPLIT_MODE": split,
                                        "SELECTOR_MODE": selector,
                                        "CORRELATION_NOISE_PCT": noise,
                                        "EXPERIMENT_SEED": seed,
                                        "NUM_ROUNDS": 50,
                                        "USE_LOGICAL_SCHEDULING": True,
                                    },
                                    run_tag,
                                )
                                done += 1
                                print(f"[suite] {done}/{total} N={n} m={m} split={split} labels={labels_per_client} selector={selector} noise={noise} seed={seed} rc={code} acc={acc}", flush=True)
                                row = {
                                    "N": n,
                                    "m": m,
                                    "split_mode": split,
                                    "labels_per_client": labels_per_client,
                                    "selector": selector,
                                    "noise_pct": noise,
                                    "seed": seed,
                                    "return_code": code,
                                    "awpsp_accuracy_last": acc,
                                    "stdout_tail": out_tail.replace("\n", "\\n"),
                                    "stderr_tail": err_tail.replace("\n", "\\n"),
                                    "metrics_path": metrics_path,
                                    "final_metrics_path": final_path,
                                }
                                rows.append(row)
                                with OUT.open("a", newline="") as f:
                                    writer = csv.DictWriter(f, fieldnames=result_fields)
                                    writer.writerow(row)

    grouped = {}
    for r in rows:
        key = (r["N"], r["m"], r["split_mode"], r["labels_per_client"], r["selector"], r["noise_pct"])
        grouped.setdefault(key, []).append(r["awpsp_accuracy_last"])

    with SUMMARY.open("w", newline="") as f:
        fieldnames = ["N", "m", "split_mode", "labels_per_client", "selector", "noise_pct", "mean_awpsp_last", "ci95"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (n, m, split, labels_per_client, selector, noise), vals in grouped.items():
            clean = [v for v in vals if isinstance(v, (float, int))]
            mean, ci = confidence_interval(clean)
            writer.writerow(
                {
                    "N": n,
                    "m": m,
                    "split_mode": split,
                    "labels_per_client": labels_per_client,
                    "selector": selector,
                    "noise_pct": noise,
                    "mean_awpsp_last": mean,
                    "ci95": ci,
                }
            )


if __name__ == "__main__":
    run_suite()
    print(f"[suite] wrote {OUT} and {SUMMARY} (per-run logs under {RUN_DIR})", flush=True)
