import os, json
from langsmith import Client
from datetime import datetime

client = Client()
run_id = "a3b0b5cb-ab8f-4267-bfbe-37dc8713dcb0"
run = client.read_run(run_id)

print("=== RUN INFO ===")
print(f"ID: {run.id}")
print(f"Name: {run.name}")
print(f"Run type: {run.run_type}")
print(f"Status: {'Error' if run.error else 'Success'}")
if run.start_time and run.end_time:
    dur = (run.end_time - run.start_time).total_seconds()
    print(f"Duration: {dur:.2f}s")
print(f"Tags: {run.tags}")
print(f"Parent ID: {run.parent_run_id}")

print("\n=== INPUT ===")
inp = run.inputs or {}
print(json.dumps(inp, indent=2, default=str)[:2000])

print("\n=== OUTPUT ===")
out = run.outputs or {}
print(json.dumps(out, indent=2, default=str)[:2000])

if run.error:
    print("\n=== ERROR ===")
    print(run.error[:1000])

print("\n=== EXTRA ===")
if run.extra:
    print(json.dumps(run.extra, indent=2, default=str)[:1000])

# Fetch child runs
print("\n=== CHILD RUNS ===")
try:
    children = list(client.list_runs(parent_run_id=run_id))
    print(f"Total children: {len(children)}")
    for c in children:
        dur = ""
        if c.start_time and c.end_time:
            d = (c.end_time - c.start_time).total_seconds()
            dur = f" [{d:.2f}s]"
        err = " ERROR" if c.error else ""
        out_preview = ""
        if c.outputs:
            out_preview = " -> " + json.dumps(c.outputs, default=str)[:100]
        inp_preview = ""
        if c.inputs:
            inp_preview = json.dumps(c.inputs, default=str)[:100]
        print(f"  - {c.name} ({c.run_type}){dur}{err}")
        print(f"    Input: {inp_preview}")
        print(f"    Output: {out_preview}")
except Exception as e:
    print(f"Error fetching children: {e}")
