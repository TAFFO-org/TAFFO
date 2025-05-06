#!/usr/bin/env python

import sys
import subprocess
from pathlib import Path

def main():
    runner_path = Path.cwd().parent / "test-runner.py"
    test_dir = Path.cwd()

    # Build the command:
    #  - use the same Python interpreter (sys.executable)
    #  - point --tests-dir at the test_dir
    #  - forward all other CLI args
    cmd = [
        sys.executable,
        str(runner_path),
        "-tests-dir", str(test_dir),
        "-common-args", "-O3 -DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS -DPOLYBENCH_STACK_ARRAYS -DCONF_GOOD -DMEDIUM_DATASET "
                        f"{test_dir}/utilities/polybench.c -I{test_dir} -I{test_dir}/utilities",
        *sys.argv[1:]
    ]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
