import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ISO_FMT_HINT = "YYYY-MM-DDTHH:MM:SS.ssssss"

def parse_iso(ts: str) -> datetime:
    # Permite timestamps com/sem microssegundos
    # e sem timezone explícito (assume UTC).
    try:
        dt = datetime.fromisoformat(ts)
    except Exception as e:
        raise ValueError(f"Timestamp inválido '{ts}' (esperado ISO 8601 como {ISO_FMT_HINT})") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

@dataclass
class PhaseMetrics:
    started: datetime | None = None
    finished: datetime | None = None
    duration_sec: float | None = None
    success: bool | None = None
    exit_code: int | None = None
    mem_max_mb: float | None = None
    mem_mean_mb: float | None = None
    cpu_path: str | None = None
    mem_path: str | None = None

@dataclass
class JobRecord:
    job_id: str
    family: str | None = None
    phases: dict = field(default_factory=lambda: {"build": PhaseMetrics(), "solve": PhaseMetrics()})
    # agregado opcional:
    model_cleanup_mb: float = 0.0

def load_events(runs_dir: Path):
    for f in runs_dir.glob("*.jsonl"):
        with f.open("r", encoding="utf-8") as fh:
            for i, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"[WARN] JSON inválido em {f.name}:{i}: {e}", file=sys.stderr)
                    continue
                yield f.name, evt

def build_job_index(events):
    jobs: dict[str, JobRecord] = {}
    all_started_ts = []
    all_finished_ts = []
    total_cleanup_mb = 0.0

    for fname, e in events:
        # Campos obrigatórios (com tolerância a logs antigos):
        ts = parse_iso(e["ts"])
        job_id = e.get("job_id") or e.get("meta", {}).get("job_id")
        if not job_id:
            print(f"[WARN] Evento sem job_id em {fname}: {e}", file=sys.stderr)
            continue

        family = e.get("family")
        phase = e.get("phase")
        kind = e.get("kind")
        state = e.get("state_at_write")
        meta = e.get("meta", {}) or {}

        if job_id not in jobs:
            jobs[job_id] = JobRecord(job_id=job_id)
        jr = jobs[job_id]
        if family:
            jr.family = family

        if phase not in jr.phases:
            jr.phases[phase] = PhaseMetrics()
        pm: PhaseMetrics = jr.phases[phase]

        if kind == "started":
            pm.started = ts
            all_started_ts.append(ts)
        elif kind == "finished":
            pm.finished = ts
            all_finished_ts.append(ts)
            # métricas principais
            pm.duration_sec = meta.get("duration_sec")
            pm.success = bool(meta.get("success", meta.get("ok")))
            pm.exit_code = meta.get("exit_code")
            metrics = meta.get("metrics") or {}
            pm.mem_max_mb = metrics.get("max")
            pm.mem_mean_mb = metrics.get("mean")
            pm.cpu_path = metrics.get("cpu")
            pm.mem_path = metrics.get("memory")

            # cleanup (só deve existir em solve ok)
            mc = meta.get("model_cleanup") or {}
            if mc.get("success") and mc.get("file_existed"):
                size = mc.get("file_size_mb")
                if isinstance(size, (int, float)):
                    jr.model_cleanup_mb += float(size)
                    total_cleanup_mb += float(size)

    return jobs, all_started_ts, all_finished_ts, total_cleanup_mb

def safe_min(xs):
    return min(xs) if xs else None

def safe_max(xs):
    return max(xs) if xs else None

def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def weighted_mean(pairs):
    # pairs: iterable de (valor, peso)
    num = 0.0
    den = 0.0
    for v, w in pairs:
        if v is None or w is None:
            continue
        num += v * w
        den += w
    return (num / den) if den > 0 else None

def summarize(jobs: dict[str, JobRecord], all_started_ts, all_finished_ts):
    # Makespan
    t0 = safe_min(all_started_ts)
    t1 = safe_max(all_finished_ts)
    makespan_sec = (t1 - t0).total_seconds() if (t0 and t1) else None

    # Throughput: jobs solve finalizados com sucesso por hora no makespan
    solved_ok = 0
    for jr in jobs.values():
        solve = jr.phases.get("solve")
        if solve and solve.finished and solve.success and (solve.exit_code == 0 or solve.exit_code is None):
            solved_ok += 1
    throughput = None
    if makespan_sec and makespan_sec > 0:
        throughput = solved_ok / (makespan_sec / 3600.0)

    # Memória global
    mem_max_global = None
    mem_max_candidates = []
    mem_mean_weighted = None
    mean_pairs = []  # (phase_mean, phase_duration)

    for jr in jobs.values():
        for ph in ("build", "solve"):
            pm: PhaseMetrics = jr.phases.get(ph)
            if not pm:
                continue
            if pm.mem_max_mb is not None:
                mem_max_candidates.append(pm.mem_max_mb)
            if pm.mem_mean_mb is not None and pm.duration_sec:
                mean_pairs.append((pm.mem_mean_mb, pm.duration_sec))

    if mem_max_candidates:
        mem_max_global = max(mem_max_candidates)
    mem_mean_weighted = weighted_mean(mean_pairs)

    # Espera solve após build (proxy)
    waits = []
    for jr in jobs.values():
        b = jr.phases.get("build")
        s = jr.phases.get("solve")
        if b and b.finished and s and s.started:
            w = (s.started - b.finished).total_seconds()
            if w >= 0:
                waits.append(w)

    wait_avg = mean(waits) if waits else None
    wait_p50 = None
    wait_p95 = None
    if waits:
        sw = sorted(waits)
        def pct(p):
            idx = max(0, min(len(sw)-1, int(round(p*(len(sw)-1)))))
            return sw[idx]
        wait_p50 = pct(0.50)
        wait_p95 = pct(0.95)

    # Contagens
    total_jobs = len(jobs)
    jobs_with_build = sum(1 for j in jobs.values() if j.phases.get("build").finished)
    jobs_with_solve = sum(1 for j in jobs.values() if j.phases.get("solve").finished)
    jobs_failed = sum(
        1 for j in jobs.values()
        if any((ph.finished and (ph.success is False or (ph.exit_code not in (None, 0))))
               for ph in j.phases.values())
    )

    return {
        "t0": t0,
        "t1": t1,
        "makespan_sec": makespan_sec,
        "throughput_jobs_per_hour": throughput,
        "mem_max_mb": mem_max_global,
        "mem_mean_mb_weighted": mem_mean_weighted,
        "wait_solve_after_build_avg_sec": wait_avg,
        "wait_solve_after_build_p50_sec": wait_p50,
        "wait_solve_after_build_p95_sec": wait_p95,
        "total_jobs": total_jobs,
        "jobs_with_build_finished": jobs_with_build,
        "jobs_with_solve_finished": jobs_with_solve,
        "jobs_failed": jobs_failed,
    }

def group_by_family(jobs: dict[str, JobRecord]):
    fam = defaultdict(list)
    for j in jobs.values():
        fam[j.family or "UNKNOWN"].append(j)
    out = []
    for family, jlist in fam.items():
        # métricas simples por família
        solved_ok = sum(1 for jr in jlist
                        if jr.phases.get("solve") and jr.phases["solve"].finished and jr.phases["solve"].success)
        durations = [ (jr.phases["build"].duration_sec or 0) + (jr.phases["solve"].duration_sec or 0)
                      for jr in jlist
                      if (jr.phases.get("build") and jr.phases.get("solve")
                          and jr.phases["solve"].finished)]
        avg_total = mean(durations) if durations else None
        out.append({
            "family": family,
            "jobs": len(jlist),
            "solved_ok": solved_ok,
            "avg_job_total_duration_sec": avg_total
        })
    out.sort(key=lambda d: d["family"])
    return out

def print_summary(summary: dict, total_cleanup_mb: float, families: list[dict]):
    def fmt_ts(dt: datetime | None):
        return dt.isoformat() if dt else "-"

    print("\n=== Aura Orchestrator — Experimental Summary ===")
    print(f"Window start (t0): {fmt_ts(summary['t0'])}")
    print(f"Window end   (t1): {fmt_ts(summary['t1'])}")
    print(f"Makespan           : {format_duration(summary['makespan_sec'])}")
    if summary['throughput_jobs_per_hour'] is not None:
        print(f"Throughput         : {summary['throughput_jobs_per_hour']:.3f} jobs/h")
    else:
        print("Throughput         : -")

    print(f"Memory (max)       : {summary['mem_max_mb']:.2f} MiB" if summary['mem_max_mb'] is not None else "Memory (max)       : -")
    print(f"Memory (mean, wgt) : {summary['mem_mean_mb_weighted']:.2f} MiB" if summary['mem_mean_mb_weighted'] is not None else "Memory (mean, wgt) : -")

    if summary['wait_solve_after_build_avg_sec'] is not None:
        print(f"Solve wait after Build (avg): {format_duration(summary['wait_solve_after_build_avg_sec'])}")
        print(f"Solve wait after Build (p50): {format_duration(summary['wait_solve_after_build_p50_sec'])}")
        print(f"Solve wait after Build (p95): {format_duration(summary['wait_solve_after_build_p95_sec'])}")
    else:
        print("Solve wait after Build      : -")

    print("\nCounts:")
    print(f"  Jobs (total)              : {summary['total_jobs']}")
    print(f"  Jobs with Build finished  : {summary['jobs_with_build_finished']}")
    print(f"  Jobs with Solve finished  : {summary['jobs_with_solve_finished']}")
    print(f"  Jobs failed (any phase)   : {summary['jobs_failed']}")
    print(f"  Model cleanup freed (MB)  : {total_cleanup_mb:.2f}")

    if families:
        print("\nPer-family snapshot:")
        print("  family, jobs, solved_ok, avg_job_total_duration_sec")
        for f in families:
            avg = f['avg_job_total_duration_sec']
            avg_fmt = f"{avg:.3f}" if avg is not None else "-"
            print(f"  {f['family']}, {f['jobs']}, {f['solved_ok']}, {avg_fmt}")

def maybe_write_csv(out_csv: Path | None, jobs: dict[str, JobRecord]):
    if not out_csv:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "job_id","family",
            "build_started","build_finished","build_duration_sec","build_success","build_exit_code","build_mem_max_mb","build_mem_mean_mb",
            "solve_started","solve_finished","solve_duration_sec","solve_success","solve_exit_code","solve_mem_max_mb","solve_mem_mean_mb",
            "solve_wait_after_build_sec","model_cleanup_mb"
        ])
        for jr in jobs.values():
            b = jr.phases.get("build") or PhaseMetrics()
            s = jr.phases.get("solve") or PhaseMetrics()
            wait = None
            if b.finished and s.started:
                d = (s.started - b.finished).total_seconds()
                if d >= 0:
                    wait = d
            w.writerow([
                jr.job_id, jr.family or "",
                b.started.isoformat() if b.started else "",
                b.finished.isoformat() if b.finished else "",
                f"{b.duration_sec:.6f}" if b.duration_sec is not None else "",
                "" if b.success is None else int(bool(b.success)),
                "" if b.exit_code is None else b.exit_code,
                "" if b.mem_max_mb is None else f"{b.mem_max_mb:.6f}",
                "" if b.mem_mean_mb is None else f"{b.mem_mean_mb:.6f}",
                s.started.isoformat() if s.started else "",
                s.finished.isoformat() if s.finished else "",
                f"{s.duration_sec:.6f}" if s.duration_sec is not None else "",
                "" if s.success is None else int(bool(s.success)),
                "" if s.exit_code is None else s.exit_code,
                "" if s.mem_max_mb is None else f"{s.mem_max_mb:.6f}",
                "" if s.mem_mean_mb is None else f"{s.mem_mean_mb:.6f}",
                "" if wait is None else f"{wait:.6f}",
                f"{jr.model_cleanup_mb:.6f}",
            ])

def main():
    ap = argparse.ArgumentParser(description="Summarize Aura Orchestrator runs.")
    ap.add_argument("--runs-dir", default="./orchestrator_runs", help="Directory with *.jsonl (default: ./orchestrator_runs)")
    ap.add_argument("--csv", type=Path, default=None, help="Optional path to write per-job CSV")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"[ERR] Runs dir not found: {runs_dir}", file=sys.stderr)
        sys.exit(2)

    events = list(load_events(runs_dir))
    if not events:
        print(f"[WARN] No events found in {runs_dir}", file=sys.stderr)
        sys.exit(0)

    jobs, all_started_ts, all_finished_ts, total_cleanup_mb = build_job_index(events)
    summary = summarize(jobs, all_started_ts, all_finished_ts)
    families = group_by_family(jobs)

    print_summary(summary, total_cleanup_mb, families)
    maybe_write_csv(args.csv, jobs)

if __name__ == "__main__":
    main()
