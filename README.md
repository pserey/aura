# Tools to evaluate the AURA Orchestrator

This repository contains tools for analyzing optimization runs from the Aura orchestrator and comparing them with serial executions.

## Scripts Overview

### 1. `scripts/summarize_runs.py` - Orchestrated Runs Analysis

Analyzes parallel optimization runs executed by the Aura orchestrator.

**Input Format:**
- Directory with `*.jsonl` files containing job events
- Corresponding `*_build_metrics.csv` and `*_solve_metrics.csv` files with CPU/memory metrics

**Usage:**
```bash
python scripts/summarize_runs.py --runs-dir /path/to/orchestrator_runs [--csv output.csv]
```

**Output:**
- Makespan and throughput
- Overall resource utilization (CPU and Memory)
- Per-phase statistics (Build and Solve):
  - Average and sum of durations
  - Parallelism metrics
  - CPU and memory consumption
- Wait times between phases
- Top 3 families by duration, memory, and CPU usage

**Example:**
```bash
python scripts/summarize_runs.py --runs-dir ~/data/orchestrator_runs --csv results.csv
```

### 2. `scripts/summarize_serial.py` - Serial Execution Analysis

Analyzes sequential optimization runs without orchestration.

**Input Format:**
- Directory structure: `results_serial/{family}/output/`
- Required files per family:
  - `error.log` - Optimizer output with timestamps and metrics
  - `cpu_output.csv` - CPU utilization over time
  - `memory_output.csv` - Memory usage over time

**Usage:**
```bash
python scripts/summarize_serial.py --results-dir /path/to/results_serial [--csv output.csv]
```

**Output:**
- Makespan and throughput
- Overall resource utilization
- Per-family average duration
- Top 3 families by duration, memory, and CPU usage

**Example:**
```bash
python scripts/summarize_serial.py --results-dir ~/data/results_serial --csv serial_results.csv
```

### 3. `scripts/compare_runs.py` - Orchestrated vs Serial Comparison

Compares orchestrated and serial executions side-by-side.

**Usage:**
```bash
python scripts/compare_runs.py \
  --orchestrated-dir /path/to/orchestrator_runs \
  --serial-dir /path/to/results_serial
```

**Output:**
- Time metrics comparison (makespan, throughput)
- Resource utilization comparison (CPU, memory)
- Speedup factor and time savings
- Efficiency analysis
- Phase breakdown for orchestrated runs

**Example:**
```bash
python scripts/compare_runs.py \
  --orchestrated-dir ~/data/orchestrator_runs \
  --serial-dir ~/data/results_serial
```

## Requirements

```bash
pip install pandas
```

## Key Metrics Explained

### Makespan
Total wall-clock time from first job start to last job end.

### Throughput
Number of jobs/families processed per hour.

### Parallelism
Average number of jobs running simultaneously. Calculated as:
```
parallelism = (sum of all job durations) / makespan
```

### CPU/Memory Metrics
- **Max**: Peak usage across all jobs/families
- **Mean (weighted)**: Average usage weighted by job duration
- **Min**: Minimum usage observed

### Speedup
Ratio of serial makespan to orchestrated makespan:
```
speedup = serial_makespan / orchestrated_makespan
```

## File Formats

### Orchestrated Runs (JSONL)
```json
{"ts": "2025-10-14T03:45:16.135807", "job_id": "...", "family": "c4", "phase": "build", "kind": "started", ...}
{"ts": "2025-10-14T03:46:55.096031", "job_id": "...", "family": "c4", "phase": "build", "kind": "finished", ...}
```

### Metrics CSV (Orchestrated)
```csv
timestamp,container_id,cpu_percent,memory_mb,elapsed_seconds
2025-10-14T03:46:57.714534,7b0bb6b2c215,0.0,16.83,0.0
2025-10-14T03:46:59.176781,7b0bb6b2c215,78.25,203.2,1.46
```

### Error Log (Serial)
```
I0000 00:00:1727997357.225024       9 aws_model.cpp:168] Number of variables = 175220
I0000 00:00:1727997772.954794       9 aws_model.cpp:202] Starting optimization
I0000 00:00:1727998226.892440       9 aws_model.cpp:204] End of optimization
I0000 00:00:1727998226.892509       9 aws_model.cpp:210] Problem solved in 869730 millisseconds
```

### CPU Output CSV (Serial)
```csv
TIMESTAMP, USER, NICE, SYS, IOWAIT, IRQ, SOFT, STEAL, GUEST, GNICE, IDLE
1727997357, 12.11, 0.00, 0.50, 0.00, 0.00, 0.12, 0.00, 0.00, 0.00, 87.27
```

### Memory Output CSV (Serial)
```csv
TIMESTAMP,TOTAL,USED,FREE,SHARED,BUFFER/CACHE,AVAILABLE
1727997357,32856456,443204,31777692,8752,635560,32016064
```

## Example Output

### Orchestrated Run Summary
```
=== Aura Orchestrator — Experimental Summary ===
Makespan           : 00:56:04.635
Throughput         : 73.827 jobs/h

--- Build Phase ---
Duration (per job avg)   : 00:01:58.149
Duration (sum all jobs)  : 02:15:52.310
Avg parallelism          : 2.42x
CPU (max)                : 101.69%

--- Solve Phase ---
Duration (per job avg)   : 00:04:20.350
Duration (sum all jobs)  : 04:59:24.121
Avg parallelism          : 5.34x
CPU (max)                : 306.92%
```

### Comparison Output
```
🚀 Overall Speedup: 5.81x faster with orchestrator
⏱️  Time Saved: 04:29:36.509 (82.8% reduction)
💻 CPU Utilization Efficiency: 47.38x
```

## Tips

1. **CSV Output**: Use `--csv` to generate detailed per-job/family data for further analysis
2. **Large Datasets**: The scripts handle large numbers of families efficiently
3. **Resource Analysis**: Compare top consumers to identify optimization opportunities
4. **Parallelism Tuning**: Use parallelism metrics to adjust orchestrator concurrency settings

## Contributing

For questions or issues, please contact the development team.
