#!/usr/bin/python3

import math

float_op_count = 0
by_opcode_count = {
    'fmul': 0,
    'fdiv': 0,
    'fadd': 0,
    'fsub': 0,
}
float_op_abs_max = 0
by_opcode_abs_max = {
    'fmul': 0,
    'fdiv': 0,
    'fadd': 0,
    'fsub': 0,
}


with open("/home/denisovlev/Projects/TAFFO/test/polybench-riscv/build_stats/1/2mm/2mm.instrumented.trace") as f:
    for line in f:
        line = line.strip()
        parts = line.split(" ")
        if len(parts) == 0 or parts[0] != "TAFFO_TRACE": continue
        value = float.fromhex(parts[2])
        opcode = parts[4]
        # print(opcode, value)
        if opcode == 'fmul' or opcode == 'fdiv' or opcode == 'fadd' or opcode == 'fsub':
            float_op_count += 1
            by_opcode_count[opcode] += 1
            float_op_abs_max = max(float_op_abs_max, abs(value))
            by_opcode_abs_max[opcode] = max(by_opcode_abs_max[opcode], abs(value))


logn = math.ceil(math.log2(float_op_count))
log_max_value = math.ceil(math.log2(float_op_abs_max))
err_float16 = logn - 7 + log_max_value
err_float19 = logn - 10 + log_max_value
err_float24 = logn - 15 + log_max_value
err_float32 = logn - 23 + log_max_value

print(by_opcode_count)
print(by_opcode_abs_max)
print("float_ops_count=", float_op_count)
print("max_value=", float_op_abs_max)
print("log(n) =", logn)
print("log(max_value) =", log_max_value)
print("log(err_float16) =", err_float16)
print("log(err_float19) =", err_float19)
print("log(err_float24) =", err_float24)
print("log(err_float32) =", err_float32)
