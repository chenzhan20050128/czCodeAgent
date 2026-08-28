# Demo fixture: buggy_calculator

A deterministic, repeatable end-to-end target for the real model.

## The bug

`calculator.py` has one wrong line: `subtract(a, b)` returns `a + b` instead of
`a - b`. The test suite fails on `test_subtract` until that line is fixed.

## Run the failing test first

```bash
cd demo/buggy_calculator
python3 -m unittest -v
```

Expected before the fix: `test_subtract` fails, exit code 1.

## Let the agent fix it

Copy the fixture to a temporary directory so the repository baseline stays
clean, then run the agent there:

```bash
cp -r demo/buggy_calculator /tmp/mca_demo
cd /tmp/mca_demo
mca "The subtract function is wrong. Fix calculator.py so 'python3 -m unittest' passes."
```

The agent should read `calculator.py`, request a `write_file`/`edit_file`
change (you approve the diff), run the tests itself, and finish once the exit
code is 0.

## Expected patch

```diff
-    return a + b  # BUG: should be a - b
+    return a - b
```

## Expected test command result after the fix

```bash
python3 -m unittest -v   # 3 tests, exit code 0
```
