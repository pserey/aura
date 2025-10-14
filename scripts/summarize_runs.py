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
    mem_min_mb: float | None = None
    cpu_max_pct: float | None = None
    cpu_mean_pct: float | None = None
    cpu_min_pct: float | None = None
    samples_count: int | None = None

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

def load_metrics_csv(csv_path: Path) -> dict:
    """Lê CSV de métricas e retorna estatísticas calculadas."""
    if not csv_path.exists():
        return {}
    
    cpu_values = []
    mem_values = []
    
    try:
        with csv_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    cpu = float(row["cpu_percent"])
                    mem = float(row["memory_mb"])
                    cpu_values.append(cpu)
                    mem_values.append(mem)
                except (ValueError, KeyError):
                    continue
    except Exception as e:
        print(f"[WARN] Erro ao ler {csv_path.name}: {e}", file=sys.stderr)
        return {}
    
    if not cpu_values or not mem_values:
        return {}
    
    return {
        "cpu_max": max(cpu_values),
        "cpu_mean": mean(cpu_values),
        "cpu_min": min(cpu_values),
        "mem_max": max(mem_values),
        "mem_mean": mean(mem_values),
        "mem_min": min(mem_values),
        "samples": len(cpu_values),
    }

def build_job_index(events, runs_dir: Path):
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
            
            # Tentar ler métricas de streaming_metrics (novo formato no JSON)
            streaming = meta.get("streaming_metrics", {})
            if streaming:
                cpu_metrics = streaming.get("cpu", {})
                mem_metrics = streaming.get("memory", {})
                collection = streaming.get("collection", {})
                
                pm.cpu_max_pct = cpu_metrics.get("max")
                pm.cpu_mean_pct = cpu_metrics.get("mean")
                pm.cpu_min_pct = cpu_metrics.get("min")
                pm.mem_max_mb = mem_metrics.get("max")
                pm.mem_mean_mb = mem_metrics.get("mean")
                pm.mem_min_mb = mem_metrics.get("min")
                pm.samples_count = collection.get("samples_count")
            else:
                # Fallback para formato antigo (metrics simples)
                metrics = meta.get("metrics") or {}
                pm.mem_max_mb = metrics.get("max")
                pm.mem_mean_mb = metrics.get("mean")

            # Tentar carregar métricas dos CSVs (nova estrutura de arquivos)
            csv_path = runs_dir / f"{job_id}_{phase}_metrics.csv"
            csv_metrics = load_metrics_csv(csv_path)
            if csv_metrics:
                # CSV tem precedência sobre JSON quando disponível
                pm.cpu_max_pct = csv_metrics.get("cpu_max")
                pm.cpu_mean_pct = csv_metrics.get("cpu_mean")
                pm.cpu_min_pct = csv_metrics.get("cpu_min")
                pm.mem_max_mb = csv_metrics.get("mem_max")
                pm.mem_mean_mb = csv_metrics.get("mem_mean")
                pm.mem_min_mb = csv_metrics.get("mem_min")
                pm.samples_count = csv_metrics.get("samples")

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

    # Métricas por fase
    phase_stats = {}
    for phase_name in ("build", "solve"):
        mem_max_candidates = []
        mem_mean_pairs = []
        cpu_max_candidates = []
        cpu_mean_pairs = []
        duration_values = []
        
        for jr in jobs.values():
            pm: PhaseMetrics = jr.phases.get(phase_name)
            if not pm or not pm.finished:
                continue
            if pm.mem_max_mb is not None:
                mem_max_candidates.append(pm.mem_max_mb)
            if pm.mem_mean_mb is not None and pm.duration_sec:
                mem_mean_pairs.append((pm.mem_mean_mb, pm.duration_sec))
            if pm.cpu_max_pct is not None:
                cpu_max_candidates.append(pm.cpu_max_pct)
            if pm.cpu_mean_pct is not None and pm.duration_sec:
                cpu_mean_pairs.append((pm.cpu_mean_pct, pm.duration_sec))
            if pm.duration_sec is not None:
                duration_values.append(pm.duration_sec)
        
        duration_sum = sum(duration_values) if duration_values else None
        avg_parallelism = None
        if duration_sum and makespan_sec and makespan_sec > 0:
            avg_parallelism = duration_sum / makespan_sec
        
        phase_stats[phase_name] = {
            "mem_max_mb": max(mem_max_candidates) if mem_max_candidates else None,
            "mem_mean_mb_weighted": weighted_mean(mem_mean_pairs),
            "cpu_max_pct": max(cpu_max_candidates) if cpu_max_candidates else None,
            "cpu_mean_pct_weighted": weighted_mean(cpu_mean_pairs),
            "duration_mean_sec": mean(duration_values) if duration_values else None,
            "duration_sum_sec": duration_sum,
            "avg_parallelism": avg_parallelism,
        }

    # Memória e CPU globais (todas as fases)
    mem_max_global = None
    mem_max_candidates = []
    mem_mean_weighted = None
    mem_mean_pairs = []  # (phase_mean, phase_duration)
    
    cpu_max_global = None
    cpu_max_candidates = []
    cpu_mean_weighted = None
    cpu_mean_pairs = []  # (phase_mean, phase_duration)

    for jr in jobs.values():
        for ph in ("build", "solve"):
            pm: PhaseMetrics = jr.phases.get(ph)
            if not pm:
                continue
            if pm.mem_max_mb is not None:
                mem_max_candidates.append(pm.mem_max_mb)
            if pm.mem_mean_mb is not None and pm.duration_sec:
                mem_mean_pairs.append((pm.mem_mean_mb, pm.duration_sec))
            if pm.cpu_max_pct is not None:
                cpu_max_candidates.append(pm.cpu_max_pct)
            if pm.cpu_mean_pct is not None and pm.duration_sec:
                cpu_mean_pairs.append((pm.cpu_mean_pct, pm.duration_sec))

    if mem_max_candidates:
        mem_max_global = max(mem_max_candidates)
    mem_mean_weighted = weighted_mean(mem_mean_pairs)
    
    if cpu_max_candidates:
        cpu_max_global = max(cpu_max_candidates)
    cpu_mean_weighted = weighted_mean(cpu_mean_pairs)

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
        "cpu_max_pct": cpu_max_global,
        "cpu_mean_pct_weighted": cpu_mean_weighted,
        "mem_max_mb": mem_max_global,
        "mem_mean_mb_weighted": mem_mean_weighted,
        "phase_stats": phase_stats,
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
        
        durations = []
        mem_max_values = []
        mem_mean_pairs = []
        cpu_max_values = []
        cpu_mean_pairs = []
        
        for jr in jlist:
            build = jr.phases.get("build")
            solve = jr.phases.get("solve")
            
            # Durações
            if build and solve and solve.finished:
                total_dur = (build.duration_sec or 0) + (solve.duration_sec or 0)
                durations.append(total_dur)
            
            # Métricas de memória e CPU
            for phase in (build, solve):
                if not phase or not phase.finished:
                    continue
                if phase.mem_max_mb is not None:
                    mem_max_values.append(phase.mem_max_mb)
                if phase.mem_mean_mb is not None and phase.duration_sec:
                    mem_mean_pairs.append((phase.mem_mean_mb, phase.duration_sec))
                if phase.cpu_max_pct is not None:
                    cpu_max_values.append(phase.cpu_max_pct)
                if phase.cpu_mean_pct is not None and phase.duration_sec:
                    cpu_mean_pairs.append((phase.cpu_mean_pct, phase.duration_sec))
        
        avg_total = mean(durations) if durations else None
        duration_sum = sum(durations) if durations else 0
        
        out.append({
            "family": family,
            "jobs": len(jlist),
            "solved_ok": solved_ok,
            "avg_job_total_duration_sec": avg_total,
            "duration_sum_sec": duration_sum,
            "mem_max_mb": max(mem_max_values) if mem_max_values else None,
            "mem_mean_mb_weighted": weighted_mean(mem_mean_pairs),
            "cpu_max_pct": max(cpu_max_values) if cpu_max_values else None,
            "cpu_mean_pct_weighted": weighted_mean(cpu_mean_pairs),
        })
    out.sort(key=lambda d: d["family"])
    return out

def print_summary(summary: dict, total_cleanup_mb: float, families: list[dict]):
    def fmt_ts(dt: datetime | None):
        return dt.isoformat() if dt else "-"
    
    def fmt_metric(value, unit="", decimals=2):
        if value is None:
            return "-"
        return f"{value:.{decimals}f}{unit}"

    print("\n=== Aura Orchestrator — Experimental Summary ===")
    print(f"Window start (t0): {fmt_ts(summary['t0'])}")
    print(f"Window end   (t1): {fmt_ts(summary['t1'])}")
    print(f"Makespan           : {format_duration(summary['makespan_sec'])}")
    if summary['throughput_jobs_per_hour'] is not None:
        print(f"Throughput         : {summary['throughput_jobs_per_hour']:.3f} jobs/h")
    else:
        print("Throughput         : -")

    print("\n--- Overall Resources ---")
    print(f"CPU (max)          : {fmt_metric(summary['cpu_max_pct'], '%')}")
    print(f"CPU (mean, wgt)    : {fmt_metric(summary['cpu_mean_pct_weighted'], '%')}")
    print(f"Memory (max)       : {fmt_metric(summary['mem_max_mb'], ' MiB')}")
    print(f"Memory (mean, wgt) : {fmt_metric(summary['mem_mean_mb_weighted'], ' MiB')}")

    # Estatísticas por fase
    phase_stats = summary.get('phase_stats', {})
    for phase_name in ("build", "solve"):
        stats = phase_stats.get(phase_name, {})
        if not stats:
            continue
        print(f"\n--- {phase_name.capitalize()} Phase ---")
        print(f"Duration (per job avg)   : {format_duration(stats.get('duration_mean_sec'))}")
        print(f"Duration (sum all jobs)  : {format_duration(stats.get('duration_sum_sec'))}")
        if stats.get('avg_parallelism') is not None:
            print(f"Avg parallelism          : {stats['avg_parallelism']:.2f}x")
        print(f"CPU (max)                : {fmt_metric(stats.get('cpu_max_pct'), '%')}")
        print(f"CPU (mean, wgt)          : {fmt_metric(stats.get('cpu_mean_pct_weighted'), '%')}")
        print(f"Memory (max)             : {fmt_metric(stats.get('mem_max_mb'), ' MiB')}")
        print(f"Memory (mean, wgt)       : {fmt_metric(stats.get('mem_mean_mb_weighted'), ' MiB')}")

    if summary['wait_solve_after_build_avg_sec'] is not None:
        print(f"\n--- Solve Wait After Build ---")
        print(f"Average: {format_duration(summary['wait_solve_after_build_avg_sec'])}")
        print(f"P50    : {format_duration(summary['wait_solve_after_build_p50_sec'])}")
        print(f"P95    : {format_duration(summary['wait_solve_after_build_p95_sec'])}")

    print("\n--- Counts ---")
    print(f"Jobs (total)              : {summary['total_jobs']}")
    print(f"Jobs with Build finished  : {summary['jobs_with_build_finished']}")
    print(f"Jobs with Solve finished  : {summary['jobs_with_solve_finished']}")
    print(f"Jobs failed (any phase)   : {summary['jobs_failed']}")
    print(f"Families (total)          : {len(families)}")
    print(f"Model cleanup freed (MB)  : {total_cleanup_mb:.2f}")
    
    # Top 3 famílias por diferentes critérios
    if families:
        print("\n--- Top 3 Families by Sum of Job Durations ---")
        top_duration = sorted(families, key=lambda f: f['duration_sum_sec'], reverse=True)[:3]
        for i, f in enumerate(top_duration, 1):
            print(f"{i}. {f['family']:30s} | {format_duration(f['duration_sum_sec'])} sum | {f['jobs']} jobs")
        
        print("\n--- Top 3 Families by Memory Consumption ---")
        top_mem = sorted([f for f in families if f['mem_max_mb'] is not None], 
                         key=lambda f: f['mem_max_mb'], reverse=True)[:3]
        for i, f in enumerate(top_mem, 1):
            mem_max = fmt_metric(f['mem_max_mb'], ' MiB')
            mem_mean = fmt_metric(f['mem_mean_mb_weighted'], ' MiB')
            print(f"{i}. {f['family']:30s} | max: {mem_max}, mean: {mem_mean} | {f['jobs']} jobs")
        
        print("\n--- Top 3 Families by CPU Usage ---")
        top_cpu = sorted([f for f in families if f['cpu_max_pct'] is not None], 
                         key=lambda f: f['cpu_max_pct'], reverse=True)[:3]
        for i, f in enumerate(top_cpu, 1):
            cpu_max = fmt_metric(f['cpu_max_pct'], '%')
            cpu_mean = fmt_metric(f['cpu_mean_pct_weighted'], '%')
            print(f"{i}. {f['family']:30s} | max: {cpu_max}, mean: {cpu_mean} | {f['jobs']} jobs")

def maybe_write_csv(out_csv: Path | None, jobs: dict[str, JobRecord]):
    if not out_csv:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "job_id","family",
            "build_started","build_finished","build_duration_sec","build_success","build_exit_code",
            "build_cpu_max_pct","build_cpu_mean_pct","build_cpu_min_pct",
            "build_mem_max_mb","build_mem_mean_mb","build_mem_min_mb","build_samples",
            "solve_started","solve_finished","solve_duration_sec","solve_success","solve_exit_code",
            "solve_cpu_max_pct","solve_cpu_mean_pct","solve_cpu_min_pct",
            "solve_mem_max_mb","solve_mem_mean_mb","solve_mem_min_mb","solve_samples",
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
                "" if b.cpu_max_pct is None else f"{b.cpu_max_pct:.2f}",
                "" if b.cpu_mean_pct is None else f"{b.cpu_mean_pct:.2f}",
                "" if b.cpu_min_pct is None else f"{b.cpu_min_pct:.2f}",
                "" if b.mem_max_mb is None else f"{b.mem_max_mb:.2f}",
                "" if b.mem_mean_mb is None else f"{b.mem_mean_mb:.2f}",
                "" if b.mem_min_mb is None else f"{b.mem_min_mb:.2f}",
                "" if b.samples_count is None else b.samples_count,
                s.started.isoformat() if s.started else "",
                s.finished.isoformat() if s.finished else "",
                f"{s.duration_sec:.6f}" if s.duration_sec is not None else "",
                "" if s.success is None else int(bool(s.success)),
                "" if s.exit_code is None else s.exit_code,
                "" if s.cpu_max_pct is None else f"{s.cpu_max_pct:.2f}",
                "" if s.cpu_mean_pct is None else f"{s.cpu_mean_pct:.2f}",
                "" if s.cpu_min_pct is None else f"{s.cpu_min_pct:.2f}",
                "" if s.mem_max_mb is None else f"{s.mem_max_mb:.2f}",
                "" if s.mem_mean_mb is None else f"{s.mem_mean_mb:.2f}",
                "" if s.mem_min_mb is None else f"{s.mem_min_mb:.2f}",
                "" if s.samples_count is None else s.samples_count,
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

    jobs, all_started_ts, all_finished_ts, total_cleanup_mb = build_job_index(events, runs_dir)
    summary = summarize(jobs, all_started_ts, all_finished_ts)
    families = group_by_family(jobs)

    print_summary(summary, total_cleanup_mb, families)
    maybe_write_csv(args.csv, jobs)

if __name__ == "__main__":
    main()
