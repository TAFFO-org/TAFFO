#!/usr/bin/env python3
"""Run and inspect the end-to-end activation-function regression tests."""

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
ACTIVATION_NAMES = {
    "relu": ("relu", "reluf", "relul"),
    "sigmoid": ("sigmoid", "sigmoidf", "sigmoidl"),
    "tanh": ("tanh", "tanhf", "tanhl"),
    "swish": ("swish", "swishf", "swishl"),
}
LUT_BITS = 10
LUT_SIZE = 1 << LUT_BITS
SWISH_CRITICAL_POINT = -1.2784645427610737


class ValidationError(RuntimeError):
    pass


def sigmoid(value):
    """Evaluate sigmoid without overflowing for large negative inputs."""
    if value >= 0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def swish(value):
    return value * sigmoid(value)


FUNCTIONS = {
    "sigmoid": sigmoid,
    "tanh": math.tanh,
    "swish": swish,
}


EXPECTED_RANGES = {
    "relu": ((0.0, 3.0), (0.0, 3.0)),
    "sigmoid": ((sigmoid(-2.0), sigmoid(1.0)),),
    "tanh": ((math.tanh(-2.0), math.tanh(1.0)),),
    "swish": ((swish(SWISH_CRITICAL_POINT), swish(1.0)),),
}


def activation_call_pattern(activation):
    names = "|".join(re.escape(name) for name in ACTIVATION_NAMES[activation])
    return re.compile(rf"\bcall\b[^\n]*@(?:{names})(?:_clone[0-9]+)?\(")


def find_final_ir(activation):
    temp_dir = TEST_DIR / activation / "taffo_temp"
    candidates = []
    for path in temp_dir.glob(f"{activation}-taffo.*.taffotmp.ll"):
        match = re.search(r"\.([0-9]+)\.taffotmp\.ll$", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise ValidationError(f"no generated IR found for {activation}")
    return max(candidates)[1]


def check_ranges():
    print("\nVRA ranges")
    for activation, expected_ranges in EXPECTED_RANGES.items():
        path = TEST_DIR / activation / "taffo_temp" / "taffo_info_vra.json"
        if not path.exists():
            raise ValidationError(f"missing VRA results for {activation}: {path}")

        data = json.loads(path.read_text())
        call_pattern = activation_call_pattern(activation)
        actual = []
        representations = []

        for value in data.get("values", {}).values():
            representation = value.get("repr", "")
            value_range = value.get("info", {}).get("range")
            if value_range and call_pattern.search(representation):
                actual.append((value_range["min"], value_range["max"]))
                representations.append(representation)

        if len(actual) != len(expected_ranges):
            raise ValidationError(
                f"{activation}: expected {len(expected_ranges)} activation ranges, "
                f"found {len(actual)}"
            )

        unmatched = list(actual)
        for expected in expected_ranges:
            for index, candidate in enumerate(unmatched):
                if all(
                    math.isclose(a, e, rel_tol=0.0, abs_tol=1e-12)
                    for a, e in zip(candidate, expected)
                ):
                    unmatched.pop(index)
                    break
            else:
                raise ValidationError(
                    f"{activation}: expected range {expected}, found {actual}"
                )

        if activation in {"relu", "sigmoid", "swish"} and not any(
            "_clone" in representation for representation in representations
        ):
            raise ValidationError(
                f"{activation}: the expected clone-aware call was not found"
            )

        formatted = ", ".join(f"[{low:.12g}, {high:.12g}]" for low, high in actual)
        print(f"  {activation:8s} OK  {formatted}")


def parse_lut(activation, ir):
    declaration_pattern = re.compile(
        rf"@__taffo_{activation}_"
        r"(?P<input_sign>[su])(?P<input_width>[0-9]+)_fp(?P<input_frac>[0-9]+)_to_"
        r"(?P<output_sign>[su])(?P<output_width>[0-9]+)_fp(?P<output_frac>[0-9]+)_"
        r"lut1024\s*=\s*(?:internal|private) constant "
        r"\[1024 x i[0-9]+\]\s*\[(?P<entries>[^\n]+)\]"
    )
    match = declaration_pattern.search(ir)
    if not match:
        raise ValidationError(f"{activation}: no 1024-entry LUT declaration found")

    entries = [
        int(value)
        for value in re.findall(r"i[0-9]+ (-?[0-9]+)", match["entries"])
    ]
    if len(entries) != LUT_SIZE:
        raise ValidationError(
            f"{activation}: expected {LUT_SIZE} LUT entries, found {len(entries)}"
        )

    return entries, {
        "input_signed": match["input_sign"] == "s",
        "input_width": int(match["input_width"]),
        "input_frac": int(match["input_frac"]),
        "output_signed": match["output_sign"] == "s",
        "output_width": int(match["output_width"]),
        "output_frac": int(match["output_frac"]),
    }


def decode_integer(raw_value, width, signed):
    bit_pattern = raw_value % (1 << width)
    if signed and bit_pattern >= (1 << (width - 1)):
        return bit_pattern - (1 << width)
    return bit_pattern


def validate_lut(activation, ir):
    table, format_info = parse_lut(activation, ir)
    function = FUNCTIONS[activation]

    input_scale_exponent = (
        format_info["input_width"] - LUT_BITS - format_info["input_frac"]
    )
    output_scale = 1 << format_info["output_frac"]
    output_width = format_info["output_width"]
    output_signed = format_info["output_signed"]
    minimum_raw = -(1 << (output_width - 1)) if output_signed else 0
    maximum_raw = (
        (1 << (output_width - 1)) - 1
        if output_signed
        else (1 << output_width) - 1
    )

    mismatches = []
    maximum_error = 0.0

    for index, stored_pattern in enumerate(table):
        input_pattern = decode_integer(index, LUT_BITS, format_info["input_signed"])
        input_value = input_pattern * (2.0 ** input_scale_exponent)

        stored_raw = decode_integer(stored_pattern, output_width, output_signed)
        expected_raw = math.floor(function(input_value) * output_scale)
        expected_raw = max(minimum_raw, min(maximum_raw, expected_raw))

        if stored_raw != expected_raw:
            mismatches.append((index, stored_raw, expected_raw))

        stored_value = stored_raw / output_scale
        maximum_error = max(maximum_error, abs(stored_value - function(input_value)))

    if mismatches:
        first = mismatches[0]
        raise ValidationError(
            f"{activation}: {len(mismatches)} LUT mismatches; "
            f"first at index {first[0]} (stored {first[1]}, expected {first[2]})"
        )

    return maximum_error


def check_conversion():
    print("\nConversion IR")
    ir_by_activation = {}
    for activation in ACTIVATION_NAMES:
        path = find_final_ir(activation)
        ir = path.read_text()
        ir_by_activation[activation] = ir
        if activation_call_pattern(activation).search(ir):
            raise ValidationError(
                f"{activation}: an original floating-point activation call remains"
            )
        print(f"  {activation:8s} OK  original calls eliminated")

    relu_ir = ir_by_activation["relu"]
    if not re.search(r"icmp sgt i[0-9]+ [^,\n]+, 0", relu_ir):
        raise ValidationError("relu: signed comparison not found")
    if not re.search(r"select i1 [^,\n]+, i[0-9]+ [^,\n]+, i[0-9]+ 0", relu_ir):
        raise ValidationError("relu: signed select not found")
    if not re.search(r"add i[0-9]+ [^,\n]*\.u[0-9]+_[0-9]+fixp, 0", relu_ir):
        raise ValidationError("relu: unsigned identity operation not found")
    print("  relu     OK  signed clamp and unsigned identity")

    print("\nLookup tables")
    for activation in FUNCTIONS:
        ir = ir_by_activation[activation]
        required_patterns = (
            rf"%{activation}\.lut\.shifted\b",
            rf"%{activation}\.lut\.pattern\b",
            rf"trunc i[0-9]+ %{activation}\.lut\.shifted to i10",
            rf"zext i10 %{activation}\.lut\.pattern to i[0-9]+",
            rf"getelementptr \[1024 x i[0-9]+\]",
            rf"load i[0-9]+, ptr %{activation}\.lut\.gep",
        )
        for pattern in required_patterns:
            if not re.search(pattern, ir):
                raise ValidationError(
                    f"{activation}: incomplete runtime LUT lookup; missing {pattern}"
                )

        maximum_error = validate_lut(activation, ir)
        print(
            f"  {activation:8s} OK  entries=1024, mismatches=0, "
            f"max_error={maximum_error:.3e}"
        )


def run_regression_suite():
    command = [
        sys.executable,
        str(TEST_DIR / "run.py"),
        "-only",
        ",".join(ACTIVATION_NAMES),
        "-debug",
    ]
    print("Running end-to-end regression tests...", flush=True)
    subprocess.run(command, cwd=TEST_DIR, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run and validate the TAFFO activation-function tests."
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="inspect existing debug artifacts without rerunning the regression suite",
    )
    args = parser.parse_args()

    try:
        if not args.skip_run:
            run_regression_suite()
        check_ranges()
        check_conversion()
    except (
        OSError,
        ValueError,
        KeyError,
        subprocess.CalledProcessError,
        ValidationError,
    ) as error:
        print(f"\nVALIDATION FAILED: {error}", file=sys.stderr)
        return 1

    print("\nAll activation-function validation checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
