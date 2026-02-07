#!/usr/bin/env python3
import sys
import time
import re
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: monitor_results.py <log_file>")
    sys.exit(1)

log_path = Path(sys.argv[1])
if not log_path.exists():
    print(f"Log file {log_path} does not exist yet. Waiting for it...")

out_csv = log_path.parent / 'core_trm_dropout_results.csv'
# write header if not exists
if not out_csv.exists():
    out_csv.write_text('timestamp,rho,recall_at_20,mrr_at_20\n')

pattern = re.compile(r"rho=([0-9.]+):\s*R@20=([^,\n]+),\s*MRR@20=([^\n]+)")
seen = set()

with log_path.open('r', encoding='utf-8', errors='ignore') as f:
    # seek to end
    f.seek(0, 2)
    while True:
        line = f.readline()
        if not line:
            time.sleep(1.0)
            continue
        m = pattern.search(line)
        if m:
            rho = m.group(1)
            recall = m.group(2).strip()
            mrr = m.group(3).strip()
            key = (rho, recall, mrr)
            if key in seen:
                continue
            seen.add(key)
            ts = time.strftime('%Y-%m-%d %H:%M:%S')
            out_csv.write_text(out_csv.read_text() + f'{ts},{rho},{recall},{mrr}\n')
            out_csv.flush()
            print(f'Logged result: rho={rho} R@20={recall} MRR@20={mrr}')
