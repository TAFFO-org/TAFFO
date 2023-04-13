#!/usr/bin/python3

import sys
import re

min_var_lines = {}
max_var_lines = {}
min_var_vals = {}
max_var_vals = {}
def main(argv):
    input_path = argv[0]
    output_path = argv[1]

    trace_regex = re.compile('TAFFO_TRACE ([^\s]+) ([^\s]+)')

    with open(input_path) as f:
        for line in f:
            trace = trace_regex.match(line)
            if trace:
                var_name = trace.group(1)
                var_value = float.fromhex(trace.group(2))
                if var_name in min_var_vals:
                    if min_var_vals[var_name] > var_value:
                        min_var_vals[var_name] = var_value
                        min_var_lines[var_name] = line
                    if max_var_vals[var_name] < var_value:
                        max_var_vals[var_name] = var_value
                        max_var_lines[var_name] = line
                else:
                    min_var_vals[var_name] = var_value
                    max_var_vals[var_name] = var_value
                    min_var_lines[var_name] = line
                    max_var_lines[var_name] = line

    with open(output_path, 'w') as f:
        for line in min_var_lines.values():
            f.write(line)
        for line in max_var_lines.values():
            f.write(line)

if __name__ == "__main__":
    main(sys.argv[1:])
