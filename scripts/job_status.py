import argparse, json, sys, subprocess, os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

# ---------- utils ----------
def parse_iso(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def fmt_dur(s: float | None) -> str:
    if s is None: return "-"
    s = float(s); h = int(s // 3600); m = int((s % 3600) // 60); sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}"

def now_utc(): return datetime.now(timezone.utc)

# ---------- domain ----------
@dataclass
class Phase:
    started: datetime | None = None
    finished: datetime | None = None
    duration_sec: float | None = None
    success: bool | None = None
    exit_code: int | None = None

@dataclass
class Job:
    job_id: str
    family: str | None = None
    build: Phase = field(default_factory=Phase)
    solve: Phase = field(default_factory=Phase)
    last_state: str | None = None

# ---------- logs ----------
def load_events(runs_dir: Path):
    for f in runs_dir.glob("*.jsonl"):
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                try:
                    yield json.loads(line)
                except Exception as e:
                    print(f"[WARN] JSON inválido em {f.name}: {e}", file=sys.stderr)

def build_index(events):
    jobs: dict[str, Job] = {}
    for e in events:
        ts = parse_iso(e["ts"])
        job_id = e.get("job_id") or e.get("meta", {}).get("job_id")
        if not job_id: continue
        family = e.get("family")
        phase_name = e.get("phase")
        kind = e.get("kind")
        state = e.get("state_at_write")
        meta = e.get("meta") or {}

        j = jobs.setdefault(job_id, Job(job_id=job_id))
        if family: j.family = family
        if state: j.last_state = state

        phase = j.build if phase_name == "build" else j.solve if phase_name == "solve" else None
        if phase is None: continue

        if kind == "started":
            phase.started = ts
        elif kind == "finished":
            phase.finished = ts
            phase.duration_sec = meta.get("duration_sec")
            succ = meta.get("success", meta.get("ok"))
            phase.success = (None if succ is None else bool(succ))
            phase.exit_code = meta.get("exit_code")
            if phase.duration_sec is None and phase.started and phase.finished:
                phase.duration_sec = (phase.finished - phase.started).total_seconds()
    return jobs

# ---------- docker helpers ----------
def run_cmd(cmd: list[str]) -> tuple[int,str,str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", "command not found"
    except Exception as e:
        return 1, "", str(e)

def docker_ps_ids_by_image(image: str) -> list[str]:
    code, out, _ = run_cmd(["docker","ps","-q","--filter",f"ancestor={image}"])
    return [x for x in out.splitlines() if x.strip()] if code == 0 else []

def docker_inspect_json(cid: str) -> dict | None:
    code, out, _ = run_cmd(["docker","inspect",cid])
    if code != 0 or not out.strip(): return None
    try: return json.loads(out)[0]
    except Exception: return None

def docker_stats_json(cid: str) -> dict | None:
    code, out, _ = run_cmd(["docker","stats","--no-stream","--format","{{json .}}",cid])
    if code != 0 or not out.strip(): return None
    try: return json.loads(out.splitlines()[0])
    except Exception: return None

def _parse_size_to_mib(s: str) -> float | None:
    if not s: return None
    first = s.split("/",1)[0].strip()
    try:
        low = first.lower()
        if low.endswith("gib") or low.endswith("gb"): return float(first[:-3 if low.endswith('gib') else -2]) * 1024.0
        if low.endswith("mib") or low.endswith("mb"): return float(first[:-3 if low.endswith('mib') else -2])
        if low.endswith("kib") or low.endswith("kb"): return float(first[:-3 if low.endswith('kib') else -2]) / 1024.0
        return float(first)  # assume MiB
    except Exception:
        return None

def _parse_percent(s: str) -> float | None:
    try: return float(s.strip().rstrip("%"))
    except Exception: return None

@dataclass
class ContainerInfo:
    cid: str
    name: str | None
    started_at: datetime | None
    mounts: list[dict]
    path: str | None
    args: list[str] | None

@dataclass
class ContainerStat:
    cid: str
    name: str | None
    cpu_perc: float | None
    mem_usage_mib: float | None
    mem_perc: float | None

def list_containers_with_info(image: str) -> list[ContainerInfo]:
    cids = docker_ps_ids_by_image(image)
    out = []
    for cid in cids:
        j = docker_inspect_json(cid) or {}
        name = j.get("Name", "")
        state = j.get("State") or {}
        started_at = None
        if state.get("StartedAt"):
            try: started_at = parse_iso(state["StartedAt"].replace("Z","+00:00"))
            except Exception: started_at = None
        mounts = j.get("Mounts") or []
        path = (j.get("Path") or None)
        args = (j.get("Args") or None)
        out.append(ContainerInfo(cid=cid, name=(name[1:] if name.startswith("/") else name), started_at=started_at, mounts=mounts, path=path, args=args))
    return out

def get_stats_for_cids(cids: list[str]) -> dict[str, ContainerStat]:
    stats: dict[str, ContainerStat] = {}
    for cid in cids:
        st = docker_stats_json(cid) or {}
        stats[cid] = ContainerStat(
            cid=cid,
            name=st.get("Name"),
            cpu_perc=_parse_percent(st.get("CPUPerc") or ""),
            mem_usage_mib=_parse_size_to_mib(st.get("MemUsage") or ""),
            mem_perc=_parse_percent(st.get("MemPerc") or ""),
        )
    return stats

def split_all_components(path: str) -> list[str]:
    parts = []
    while True:
        path, tail = os.path.split(path)
        if tail: parts.append(tail)
        else:
            if path: parts.append(path)
            break
    return [p for p in parts if p and p not in ("/","\\")]

def infer_container_family_from_mounts(mounts: list[dict], known_families: set[str]) -> str | None:
    # considera TODOS os componentes do Source e tenta casar com uma família conhecida
    for m in mounts:
        src = (m.get("Source") or "").rstrip("/")
        if not src: continue
        comps = split_all_components(src)
        for c in comps:
            if c in known_families:
                return c
    return None

def infer_container_family_from_cmd(name: str | None, path: str | None, args: list[str] | None, known_families: set[str]) -> str | None:
    hay = []
    if name: hay.append(name)
    if path: hay.append(path)
    if args: hay.extend(args)
    text = " ".join(hay)
    for fam in sorted(known_families, key=len, reverse=True):
        if fam in text:
            return fam
    return None

# ---------- status ----------
def summarize(jobs: dict[str, Job], image: str, use_docker: bool):
    build_done = sum(1 for j in jobs.values() if j.build.finished)
    solve_done = sum(1 for j in jobs.values() if j.solve.finished)

    # famílias conhecidas a partir dos logs (melhora o matching)
    known_families = {j.family for j in jobs.values() if j.family}

    containers_info = list_containers_with_info(image) if use_docker else []
    stats = get_stats_for_cids([c.cid for c in containers_info]) if use_docker and containers_info else {}

    # pré-computa família inferida por container
    container_family: dict[str, str | None] = {}
    for ci in containers_info:
        fam = infer_container_family_from_mounts(ci.mounts, known_families)
        if not fam:
            fam = infer_container_family_from_cmd(ci.name, ci.path, ci.args, known_families)
        container_family[ci.cid] = fam

    # index por família
    family_to_containers: dict[str, list[str]] = {}
    for cid, fam in container_family.items():
        if fam:
            family_to_containers.setdefault(fam, []).append(cid)

    # monta “solving agora”
    solving_now = []
    tnow = now_utc()
    for j in jobs.values():
        if j.solve.started and not j.solve.finished:
            elapsed = (tnow - j.solve.started).total_seconds()
            fam = j.family or "UNKNOWN"

            # 1) tentativa por família
            cids = family_to_containers.get(fam, [])
            chosen = None

            # 2) se não houver matching por família, tenta **proximidade temporal**
            if not cids and containers_info:
                # pega o container mais próximo de solve.started
                best = None; best_delta = None
                for ci in containers_info:
                    if not ci.started_at: continue
                    delta = abs((ci.started_at - j.solve.started).total_seconds())
                    if (best_delta is None) or (delta < best_delta):
                        best_delta = delta; best = ci
                if best: cids = [best.cid]

            if cids:
                # escolhe o de maior memória no momento
                chosen = max(cids, key=lambda x: (stats.get(x).mem_usage_mib if stats.get(x) and stats.get(x).mem_usage_mib is not None else -1.0))

            if chosen and chosen in stats:
                st = stats[chosen]
                solving_now.append({
                    "job_id": j.job_id,
                    "family": fam,
                    "solve_started": j.solve.started.isoformat(),
                    "elapsed_sec": elapsed,
                    "docker_container": chosen,
                    "docker_name": st.name,
                    "mem_usage_mib": st.mem_usage_mib,
                    "mem_perc": st.mem_perc,
                    "cpu_perc": st.cpu_perc,
                    "containers_in_family": len(cids)
                })
            else:
                solving_now.append({
                    "job_id": j.job_id,
                    "family": fam,
                    "solve_started": j.solve.started.isoformat(),
                    "elapsed_sec": elapsed,
                    "docker_container": None,
                    "docker_name": None,
                    "mem_usage_mib": None,
                    "mem_perc": None,
                    "cpu_perc": None,
                    "containers_in_family": 0
                })

    build_durations = [j.build.duration_sec for j in jobs.values() if j.build.duration_sec is not None and j.build.finished]
    solve_durations = [j.solve.duration_sec for j in jobs.values() if j.solve.duration_sec is not None and j.solve.finished]
    build_avg = mean(build_durations) if build_durations else None
    solve_avg = mean(solve_durations) if solve_durations else None

    return {
        "total_jobs": len(jobs),
        "build_finished": build_done,
        "solve_finished": solve_done,
        "solving_now_count": len(solving_now),
        "solving_now": solving_now,
        "build_avg_duration_sec": build_avg,
        "solve_avg_duration_sec": solve_avg,
    }

def print_human(summary: dict, limit_solving: int, show_mem: bool):
    print("\n=== Aura Orchestrator — Jobs Status (with Docker mem) ===")
    print(f"Jobs (total)        : {summary['total_jobs']}")
    print(f"Build finished      : {summary['build_finished']}")
    print(f"Solve finished      : {summary['solve_finished']}")
    print(f"Solving now         : {summary['solving_now_count']}")
    print(f"Avg Build duration  : {fmt_dur(summary['build_avg_duration_sec'])}")
    print(f"Avg Solve duration  : {fmt_dur(summary['solve_avg_duration_sec'])}")

    if summary["solving_now"]:
        hdr = ["job_id","family","started_iso","elapsed"]
        if show_mem: hdr += ["mem(MiB)","mem(%)","cpu(%)","cid","name","#fam"]
        print("\nCurrently solving (up to {limit}):".format(limit=limit_solving))
        print("  " + " | ".join(h.ljust(16) for h in hdr))
        for it in summary["solving_now"][:limit_solving]:
            row = [
                it["job_id"][:16],
                (it["family"] or "")[:16],
                it["solve_started"][:19],
                fmt_dur(it["elapsed_sec"])
            ]
            if show_mem:
                mem_mib = "-" if it["mem_usage_mib"] is None else f"{it['mem_usage_mib']:.1f}"
                mem_p = "-" if it["mem_perc"] is None else f"{it['mem_perc']:.2f}"
                cpu_p = "-" if it["cpu_perc"] is None else f"{it['cpu_perc']:.2f}"
                cid = (it["docker_container"] or "-")[:12]
                name = (it["docker_name"] or "-")[:16]
                nf = str(it.get("containers_in_family", 0))
                row += [mem_mib, mem_p, cpu_p, cid, name, nf]
            print("  " + " | ".join(str(x).ljust(16) for x in row))

def main():
    ap = argparse.ArgumentParser(description="Job status from Aura logs + Docker memory for solving jobs.")
    ap.add_argument("--runs-dir", default="./orchestrator_runs", help="Dir with *.jsonl (default: ./orchestrator_runs)")
    ap.add_argument("--docker-image", default="awsome-savings:optimizer", help="Docker image filter (ancestor=)")
    ap.add_argument("--json", action="store_true", help="Print JSON instead of human-readable")
    ap.add_argument("--limit-solving", type=int, default=20)
    ap.add_argument("--no-docker", action="store_true", help="Skip Docker integration")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"[ERR] Runs dir not found: {runs_dir}", file=sys.stderr); sys.exit(2)

    evts = list(load_events(runs_dir))
    if not evts:
        print(f"[WARN] No events found in {runs_dir}", file=sys.stderr); sys.exit(0)

    jobs = build_index(evts)
    summary = summarize(jobs, image=args.docker-image if not args.no_docker else "", use_docker=not args.no_docker)

    if args.json:
        out = summary.copy()
        out["solving_now"] = [{
            **it,
            "elapsed_sec": round(it["elapsed_sec"], 3),
            "mem_usage_mib": None if it["mem_usage_mib"] is None else round(it["mem_usage_mib"], 3),
            "mem_perc": None if it["mem_perc"] is None else round(it["mem_perc"], 3),
            "cpu_perc": None if it["cpu_perc"] is None else round(it["cpu_perc"], 3),
        } for it in summary["solving_now"]]
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print_human(summary, args.limit_solving, show_mem=not args.no_docker)

if __name__ == "__main__":
    main()
