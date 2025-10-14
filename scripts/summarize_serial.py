import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

try:
    import pandas as pd
except ImportError:
    print("[ERR] pandas is required. Install with: pip install pandas", file=sys.stderr)
    sys.exit(1)

@dataclass
class FamilyResult:
    family: str
    start_ts: datetime | None = None
    end_ts: datetime | None = None
    duration_sec: float | None = None
    num_variables: int | None = None
    objective_value: float | None = None
    iterations: int | None = None
    cpu_max_pct: float | None = None
    cpu_mean_pct: float | None = None
    cpu_min_pct: float | None = None
    mem_max_mb: float | None = None
    mem_mean_mb: float | None = None
    mem_min_mb: float | None = None
    samples_count: int = 0
    success: bool = False

def parse_error_log(log_path: Path) -> dict:
    """Parse error.log to extract key information."""
    if not log_path.exists():
        return {}
    
    info = {}
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        # Extract timestamps and key metrics
        # Pattern: I0000 00:00:1727997357.225024
        ts_pattern = r"I0000 00:00:(\d+\.\d+)\s+\d+\s+aws_model\.cpp:\d+\]\s*(.+)"
        
        for match in re.finditer(ts_pattern, content):
            ts_str = match.group(1)
            message = match.group(2)
            ts = float(ts_str)
            
            if "Number of variables" in message:
                var_match = re.search(r"Number of variables = (\d+)", message)
                if var_match:
                    info["num_variables"] = int(var_match.group(1))
                info["start_ts"] = ts
            elif "Starting optimization" in message:
                info["optimization_start_ts"] = ts
            elif "End of optimization" in message:
                info["end_ts"] = ts
            elif "Objective value" in message:
                obj_match = re.search(r"Objective value = ([\d.e+]+)", message)
                if obj_match:
                    info["objective_value"] = float(obj_match.group(1))
            elif "Problem solved in" and "millisseconds" in message:
                dur_match = re.search(r"Problem solved in (\d+) millisseconds", message)
                if dur_match:
                    info["duration_ms"] = int(dur_match.group(1))
            elif "Problem solved in" and "iterations" in message:
                iter_match = re.search(r"Problem solved in (\d+) iterations", message)
                if iter_match:
                    info["iterations"] = int(iter_match.group(1))
        
        # Determine success
        if "End of optimization" in content:
            info["success"] = True
            
    except Exception as e:
        print(f"[WARN] Error parsing {log_path}: {e}", file=sys.stderr)
    
    return info

def parse_cpu_csv(csv_path: Path) -> dict:
    """Parse cpu_output.csv and calculate statistics."""
    if not csv_path.exists():
        return {}
    
    try:
        # Read CSV with pandas to handle spaces
        df = pd.read_csv(csv_path, skipinitialspace=True)
        
        # Calculate CPU usage (100 - IDLE)
        df['CPU_USED'] = 100.0 - df['IDLE']
        
        return {
            "cpu_max": df['CPU_USED'].max(),
            "cpu_mean": df['CPU_USED'].mean(),
            "cpu_min": df['CPU_USED'].min(),
            "samples": len(df),
        }
    except Exception as e:
        print(f"[WARN] Error parsing {csv_path}: {e}", file=sys.stderr)
        return {}

def parse_memory_csv(csv_path: Path) -> dict:
    """Parse memory_output.csv and calculate statistics."""
    if not csv_path.exists():
        return {}
    
    try:
        # Read CSV with pandas
        df = pd.read_csv(csv_path, skipinitialspace=True)
        
        # Convert from KB to MB
        df['USED_MB'] = df['USED'] / 1024.0
        df['TOTAL_MB'] = df['TOTAL'] / 1024.0
        
        return {
            "mem_max": df['USED_MB'].max(),
            "mem_mean": df['USED_MB'].mean(),
            "mem_min": df['USED_MB'].min(),
            "samples": len(df),
        }
    except Exception as e:
        print(f"[WARN] Error parsing {csv_path}: {e}", file=sys.stderr)
        return {}

def load_family_results(results_dir: Path) -> list[FamilyResult]:
    """Load all family results from results_serial directory."""
    results = []
    
    if not results_dir.exists():
        print(f"[ERR] Results directory not found: {results_dir}", file=sys.stderr)
        return results
    
    # Iterate through family directories
    for family_dir in sorted(results_dir.iterdir()):
        if not family_dir.is_dir():
            continue
        
        family_name = family_dir.name
        output_dir = family_dir / "output"
        
        if not output_dir.exists():
            print(f"[WARN] No output directory for family {family_name}", file=sys.stderr)
            continue
        
        fr = FamilyResult(family=family_name)
        
        # Parse error.log
        error_log = output_dir / "error.log"
        log_info = parse_error_log(error_log)
        
        if log_info:
            if "start_ts" in log_info:
                fr.start_ts = datetime.fromtimestamp(log_info["start_ts"], tz=timezone.utc)
            if "end_ts" in log_info:
                fr.end_ts = datetime.fromtimestamp(log_info["end_ts"], tz=timezone.utc)
            if "duration_ms" in log_info:
                fr.duration_sec = log_info["duration_ms"] / 1000.0
            fr.num_variables = log_info.get("num_variables")
            fr.objective_value = log_info.get("objective_value")
            fr.iterations = log_info.get("iterations")
            fr.success = log_info.get("success", False)
        
        # Parse CPU metrics
        cpu_csv = output_dir / "cpu_output.csv"
        cpu_info = parse_cpu_csv(cpu_csv)
        if cpu_info:
            fr.cpu_max_pct = cpu_info.get("cpu_max")
            fr.cpu_mean_pct = cpu_info.get("cpu_mean")
            fr.cpu_min_pct = cpu_info.get("cpu_min")
            fr.samples_count = cpu_info.get("samples", 0)
        
        # Parse memory metrics
        mem_csv = output_dir / "memory_output.csv"
        mem_info = parse_memory_csv(mem_csv)
        if mem_info:
            fr.mem_max_mb = mem_info.get("mem_max")
            fr.mem_mean_mb = mem_info.get("mem_mean")
            fr.mem_min_mb = mem_info.get("mem_min")
            # Use max samples count from CPU or memory
            fr.samples_count = max(fr.samples_count, mem_info.get("samples", 0))
        
        results.append(fr)
    
    return results

def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def fmt_metric(value, unit="", decimals=2):
    if value is None:
        return "-"
    return f"{value:.{decimals}f}{unit}"

def calculate_summary(results: list[FamilyResult]) -> dict:
    """Calculate aggregate statistics."""
    successful = [r for r in results if r.success]
    
    # Time window
    all_starts = [r.start_ts for r in results if r.start_ts]
    all_ends = [r.end_ts for r in results if r.end_ts]
    t0 = min(all_starts) if all_starts else None
    t1 = max(all_ends) if all_ends else None
    makespan_sec = (t1 - t0).total_seconds() if (t0 and t1) else None
    
    # Durations
    durations = [r.duration_sec for r in successful if r.duration_sec]
    duration_sum = sum(durations) if durations else None
    duration_mean = mean(durations) if durations else None
    
    # Throughput
    throughput = None
    if makespan_sec and makespan_sec > 0 and successful:
        throughput = len(successful) / (makespan_sec / 3600.0)
    
    # CPU stats
    cpu_values = [(r.cpu_max_pct, r.cpu_mean_pct) for r in successful 
                  if r.cpu_max_pct is not None]
    cpu_max = max([v[0] for v in cpu_values]) if cpu_values else None
    cpu_mean_overall = mean([v[1] for v in cpu_values]) if cpu_values else None
    
    # Memory stats
    mem_values = [(r.mem_max_mb, r.mem_mean_mb) for r in successful 
                  if r.mem_max_mb is not None]
    mem_max = max([v[0] for v in mem_values]) if mem_values else None
    mem_mean_overall = mean([v[1] for v in mem_values]) if mem_values else None
    
    return {
        "t0": t0,
        "t1": t1,
        "makespan_sec": makespan_sec,
        "throughput_jobs_per_hour": throughput,
        "duration_sum_sec": duration_sum,
        "duration_mean_sec": duration_mean,
        "cpu_max_pct": cpu_max,
        "cpu_mean_pct": cpu_mean_overall,
        "mem_max_mb": mem_max,
        "mem_mean_mb": mem_mean_overall,
        "total_families": len(results),
        "successful_families": len(successful),
        "failed_families": len(results) - len(successful),
    }

def print_summary(summary: dict, results: list[FamilyResult]):
    """Print summary statistics."""
    def fmt_ts(dt: datetime | None):
        return dt.isoformat() if dt else "-"
    
    print("\n=== Serial Optimization Results Summary ===")
    print(f"Window start (t0): {fmt_ts(summary['t0'])}")
    print(f"Window end   (t1): {fmt_ts(summary['t1'])}")
    print(f"Makespan           : {format_duration(summary['makespan_sec'])}")
    if summary['throughput_jobs_per_hour'] is not None:
        print(f"Throughput         : {summary['throughput_jobs_per_hour']:.3f} families/h")
    else:
        print("Throughput         : -")
    
    print("\n--- Overall Resources (Serial Execution) ---")
    print(f"Duration (per family avg)  : {format_duration(summary['duration_mean_sec'])}")
    print(f"Duration (sum all families): {format_duration(summary['duration_sum_sec'])}")
    print(f"CPU (max)                  : {fmt_metric(summary['cpu_max_pct'], '%')}")
    print(f"CPU (mean across families) : {fmt_metric(summary['cpu_mean_pct'], '%')}")
    print(f"Memory (max)               : {fmt_metric(summary['mem_max_mb'], ' MiB')}")
    print(f"Memory (mean across fam.)  : {fmt_metric(summary['mem_mean_mb'], ' MiB')}")
    
    print("\n--- Counts ---")
    print(f"Families (total)      : {summary['total_families']}")
    print(f"Families (successful) : {summary['successful_families']}")
    print(f"Families (failed)     : {summary['failed_families']}")
    
    # Top 3 by different criteria
    successful = [r for r in results if r.success]
    
    if successful:
        print("\n--- Top 3 Families by Duration ---")
        top_duration = sorted(successful, 
                             key=lambda r: r.duration_sec if r.duration_sec else 0, 
                             reverse=True)[:3]
        for i, r in enumerate(top_duration, 1):
            print(f"{i}. {r.family:30s} | {format_duration(r.duration_sec)}")
        
        print("\n--- Top 3 Families by Memory Consumption ---")
        top_mem = sorted([r for r in successful if r.mem_max_mb is not None],
                        key=lambda r: r.mem_max_mb,
                        reverse=True)[:3]
        for i, r in enumerate(top_mem, 1):
            mem_max = fmt_metric(r.mem_max_mb, ' MiB')
            mem_mean = fmt_metric(r.mem_mean_mb, ' MiB')
            print(f"{i}. {r.family:30s} | max: {mem_max}, mean: {mem_mean}")
        
        print("\n--- Top 3 Families by CPU Usage ---")
        top_cpu = sorted([r for r in successful if r.cpu_max_pct is not None],
                        key=lambda r: r.cpu_max_pct,
                        reverse=True)[:3]
        for i, r in enumerate(top_cpu, 1):
            cpu_max = fmt_metric(r.cpu_max_pct, '%')
            cpu_mean = fmt_metric(r.cpu_mean_pct, '%')
            print(f"{i}. {r.family:30s} | max: {cpu_max}, mean: {cpu_mean}")

def maybe_write_csv(out_csv: Path | None, results: list[FamilyResult]):
    """Write detailed results to CSV."""
    if not out_csv:
        return
    
    import csv
    
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "family", "success", "start_ts", "end_ts", "duration_sec",
            "num_variables", "objective_value", "iterations",
            "cpu_max_pct", "cpu_mean_pct", "cpu_min_pct",
            "mem_max_mb", "mem_mean_mb", "mem_min_mb",
            "samples_count"
        ])
        for r in results:
            w.writerow([
                r.family,
                1 if r.success else 0,
                r.start_ts.isoformat() if r.start_ts else "",
                r.end_ts.isoformat() if r.end_ts else "",
                f"{r.duration_sec:.6f}" if r.duration_sec is not None else "",
                r.num_variables or "",
                f"{r.objective_value:.6e}" if r.objective_value is not None else "",
                r.iterations or "",
                f"{r.cpu_max_pct:.2f}" if r.cpu_max_pct is not None else "",
                f"{r.cpu_mean_pct:.2f}" if r.cpu_mean_pct is not None else "",
                f"{r.cpu_min_pct:.2f}" if r.cpu_min_pct is not None else "",
                f"{r.mem_max_mb:.2f}" if r.mem_max_mb is not None else "",
                f"{r.mem_mean_mb:.2f}" if r.mem_mean_mb is not None else "",
                f"{r.mem_min_mb:.2f}" if r.mem_min_mb is not None else "",
                r.samples_count,
            ])
    print(f"\nDetailed CSV written to: {out_csv}")

def main():
    ap = argparse.ArgumentParser(description="Summarize serial optimization results.")
    ap.add_argument("--results-dir", required=True, 
                   help="Directory with family subdirectories (e.g., results_serial/)")
    ap.add_argument("--csv", type=Path, default=None, 
                   help="Optional path to write per-family CSV")
    args = ap.parse_args()
    
    results_dir = Path(args.results_dir)
    results = load_family_results(results_dir)
    
    if not results:
        print(f"[WARN] No results found in {results_dir}", file=sys.stderr)
        sys.exit(0)
    
    summary = calculate_summary(results)
    print_summary(summary, results)
    maybe_write_csv(args.csv, results)

if __name__ == "__main__":
    main()

