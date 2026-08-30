# Demo fixture: parallel_bugfix

This fixture has two independent bugs in two files. It is intended for a real
Code Mode demonstration in which the model reads both files, prepares both
edits in parallel, and runs the dependent test suite only after both succeed.

Copy it to a temporary directory before running the agent so this repository
keeps the deliberately failing baseline.

```bash
cp -R demo/parallel_bugfix /tmp/mca_parallel_demo
cd /tmp/mca_parallel_demo
python3 -m unittest -v  # two failures
mca --yolo "Use one run_code call: inspect the project, fix both independent bugs with parallel file edits, then run python3 -m unittest -v only after both edits succeed."
```

Expected fixes: `discounted_price` subtracts the discount and `slugify`
lowercases before replacing spaces. The final DAG has two independent edit
nodes joining into one Bash test node.
