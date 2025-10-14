#!/usr/bin/env python3
"""
Compare orchestrated vs serial optimization runs.
"""
import argparse
import sys
from pathlib import Path

# Import the summarization scripts
from summarize_runs import load_events, build_job_index, summarize, group_by_family
from summarize_serial import load_family_results, calculate_summary

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

def calculate_speedup(orchestrated_sec, serial_sec):
    """Calculate speedup factor."""
    if orchestrated_sec and serial_sec and orchestrated_sec > 0:
        return serial_sec / orchestrated_sec
    return None

def print_comparison(orch_summary: dict, serial_summary: dict):
    """Print side-by-side comparison."""
    
    print("\n" + "=" * 80)
    print(" " * 20 + "ORCHESTRATED vs SERIAL COMPARISON")
    print("=" * 80)
    
    # Makespan comparison
    print("\n--- Time Metrics ---")
    print(f"{'Metric':<30} {'Orchestrated':>20} {'Serial':>20} {'Speedup':>10}")
    print("-" * 80)
    
    orch_makespan = orch_summary.get('makespan_sec')
    serial_makespan = serial_summary.get('makespan_sec')
    speedup = calculate_speedup(orch_makespan, serial_makespan)
    
    print(f"{'Makespan':<30} {format_duration(orch_makespan):>20} "
          f"{format_duration(serial_makespan):>20} "
          f"{fmt_metric(speedup, 'x'):>10}")
    
    orch_throughput = orch_summary.get('throughput_jobs_per_hour')
    serial_throughput = serial_summary.get('throughput_jobs_per_hour')
    
    print(f"{'Throughput (jobs/h)':<30} {fmt_metric(orch_throughput, '', 3):>20} "
          f"{fmt_metric(serial_throughput, '', 3):>20} "
          f"{fmt_metric(orch_throughput / serial_throughput if (orch_throughput and serial_throughput) else None, 'x'):>10}")
    
    # Resource utilization
    print("\n--- Resource Utilization ---")
    print(f"{'Metric':<30} {'Orchestrated':>20} {'Serial':>20} {'Ratio':>10}")
    print("-" * 80)
    
    orch_cpu = orch_summary.get('cpu_max_pct')
    serial_cpu = serial_summary.get('cpu_max_pct')
    print(f"{'CPU Max (%)':<30} {fmt_metric(orch_cpu):>20} "
          f"{fmt_metric(serial_cpu):>20} "
          f"{fmt_metric(orch_cpu / serial_cpu if (orch_cpu and serial_cpu) else None, 'x'):>10}")
    
    orch_cpu_mean = orch_summary.get('cpu_mean_pct_weighted')
    serial_cpu_mean = serial_summary.get('cpu_mean_pct')
    print(f"{'CPU Mean (%)':<30} {fmt_metric(orch_cpu_mean):>20} "
          f"{fmt_metric(serial_cpu_mean):>20} "
          f"{fmt_metric(orch_cpu_mean / serial_cpu_mean if (orch_cpu_mean and serial_cpu_mean) else None, 'x'):>10}")
    
    orch_mem = orch_summary.get('mem_max_mb')
    serial_mem = serial_summary.get('mem_max_mb')
    print(f"{'Memory Max (MiB)':<30} {fmt_metric(orch_mem, '', 0):>20} "
          f"{fmt_metric(serial_mem, '', 0):>20} "
          f"{fmt_metric(orch_mem / serial_mem if (orch_mem and serial_mem) else None, 'x'):>10}")
    
    orch_mem_mean = orch_summary.get('mem_mean_mb_weighted')
    serial_mem_mean = serial_summary.get('mem_mean_mb')
    print(f"{'Memory Mean (MiB)':<30} {fmt_metric(orch_mem_mean, '', 0):>20} "
          f"{fmt_metric(serial_mem_mean, '', 0):>20} "
          f"{fmt_metric(orch_mem_mean / serial_mem_mean if (orch_mem_mean and serial_mem_mean) else None, 'x'):>10}")
    
    # Phase comparison (orchestrated only)
    phase_stats = orch_summary.get('phase_stats', {})
    if phase_stats:
        print("\n--- Orchestrated Phase Breakdown ---")
        print(f"{'Phase':<15} {'Avg Duration':>15} {'Sum Duration':>15} {'Parallelism':>12}")
        print("-" * 60)
        for phase_name in ("build", "solve"):
            stats = phase_stats.get(phase_name, {})
            if stats:
                print(f"{phase_name.capitalize():<15} "
                      f"{format_duration(stats.get('duration_mean_sec')):>15} "
                      f"{format_duration(stats.get('duration_sum_sec')):>15} "
                      f"{fmt_metric(stats.get('avg_parallelism'), 'x'):>12}")
    
    # Efficiency analysis
    print("\n--- Efficiency Analysis ---")
    orch_jobs = orch_summary.get('total_jobs', 0)
    serial_jobs = serial_summary.get('total_families', 0)
    print(f"Total families processed: {orch_jobs} (orchestrated) vs {serial_jobs} (serial)")
    
    if speedup:
        print(f"\n🚀 Overall Speedup: {speedup:.2f}x faster with orchestrator")
        time_saved = serial_makespan - orch_makespan if (serial_makespan and orch_makespan) else None
        if time_saved:
            print(f"⏱️  Time Saved: {format_duration(time_saved)} ({time_saved/serial_makespan*100:.1f}% reduction)")
    
    # Resource efficiency
    if orch_cpu_mean and serial_cpu_mean:
        cpu_efficiency = (orch_cpu_mean / serial_cpu_mean) * (serial_makespan / orch_makespan if (serial_makespan and orch_makespan and orch_makespan > 0) else 1)
        print(f"💻 CPU Utilization Efficiency: {cpu_efficiency:.2f}x")
    
    print("\n" + "=" * 80)

def main():
    ap = argparse.ArgumentParser(description="Compare orchestrated vs serial runs.")
    ap.add_argument("--orchestrated-dir", required=True, 
                   help="Directory with orchestrated runs (*.jsonl files)")
    ap.add_argument("--serial-dir", required=True,
                   help="Directory with serial results (family subdirectories)")
    args = ap.parse_args()
    
    orch_dir = Path(args.orchestrated_dir)
    serial_dir = Path(args.serial_dir)
    
    # Load orchestrated results
    print("Loading orchestrated results...")
    events = list(load_events(orch_dir))
    if not events:
        print(f"[ERR] No orchestrated events found in {orch_dir}", file=sys.stderr)
        sys.exit(1)
    
    orch_jobs, all_started_ts, all_finished_ts, _ = build_job_index(events, orch_dir)
    orch_summary = summarize(orch_jobs, all_started_ts, all_finished_ts)
    
    # Load serial results
    print("Loading serial results...")
    serial_results = load_family_results(serial_dir)
    if not serial_results:
        print(f"[ERR] No serial results found in {serial_dir}", file=sys.stderr)
        sys.exit(1)
    
    serial_summary = calculate_summary(serial_results)
    
    # Print comparison
    print_comparison(orch_summary, serial_summary)

if __name__ == "__main__":
    main()

