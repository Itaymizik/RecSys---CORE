#!/usr/bin/env python3
"""
Tail a log file and extract Recall@20 and MRR@20 values when they appear.
Writes a summary CSV `core_trm_full_test_metrics.csv` (appends) and prints the found values.
Usage: python scripts/wait_for_metrics.py /path/to/logfile
"""
import sys
import time
import re
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: wait_for_metrics.py /path/to/logfile [output_csv]")
    sys.exit(2)

logpath = Path(sys.argv[1])
out_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('core_trm_full_test_metrics.csv')

# Regex patterns to capture values in common RecBole output formats
pat_recall = re.compile(r"Recall@20['\"]?\s*[:=]?\s*([0-9.efE+\-]+)")
pat_mrr = re.compile(r"MRR@20['\"]?\s*[:=]?\s*([0-9.efE+\-]+)")
# Also handle dict-like printouts: 'Recall@20': 0.1234
pat_recall2 = re.compile(r"'Recall@20'\s*:\s*([0-9.efE+\-]+)")
pat_mrr2 = re.compile(r"'MRR@20'\s*:\s*([0-9.efE+\-]+)")

recall = None
mrr = None

# Wait for file to exist
while not logpath.exists():
    time.sleep(1)

with open(logpath, 'r', errors='ignore') as f:
    # Seek to end to only watch new output
    f.seek(0, 2)
    print(f"Watching {logpath} for Recall@20 and MRR@20...")
    try:
        while True:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue
            line = line.strip()
            # search for recall
            if recall is None:
                m = pat_recall.search(line) or pat_recall2.search(line)
                if m:
                    try:
                        recall = float(m.group(1))
                        print(f"Found Recall@20 = {recall}")
                    except Exception:
                        pass
            if mrr is None:
                m = pat_mrr.search(line) or pat_mrr2.search(line)
                if m:
                    try:
                        mrr = float(m.group(1))
                        print(f"Found MRR@20 = {mrr}")
                    except Exception:
                        pass
            # Some RecBole prints full dict in one line; try to extract both
            if (recall is None or mrr is None) and ("Recall@20" in line or "MRR@20" in line):
                # try dict-style
                m1 = pat_recall2.search(line)
                m2 = pat_mrr2.search(line)
                if m1 and recall is None:
                    recall = float(m1.group(1))
                    print(f"Found Recall@20 = {recall}")
                if m2 and mrr is None:
                    mrr = float(m2.group(1))
                    print(f"Found MRR@20 = {mrr}")

            if recall is not None and mrr is not None:
                t = time.strftime('%Y-%m-%d %H:%M:%S')
                out_line = f"{t},{logpath.name},{recall},{mrr}\n"
                header = 'timestamp,logfile,Recall@20,MRR@20\n'
                if not out_csv.exists():
                    out_csv.write_text(header)
                with out_csv.open('a') as o:
                    o.write(out_line)
                print(f"Wrote metrics to {out_csv}: Recall@20={recall}, MRR@20={mrr}")
                break
    except KeyboardInterrupt:
        print('Interrupted, exiting')

print('Monitor finished')
