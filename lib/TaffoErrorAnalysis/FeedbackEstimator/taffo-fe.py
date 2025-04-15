#!/usr/bin/env python3
import argparse
import pickle
import sys
from copy import copy
from pathlib import Path


def nullfunc(*args, **kwargs):
    pass
vprint = nullfunc


def parse_perfest(fpath):
    text = fpath.read_text()
    return float(text.strip())


def parse_errorprop(fpath):
    text = fpath.read_text().splitlines()
    errors = []
    for line in text:
        if not line.startswith('Computed error'):
            continue
        parts = line.strip().split(' ')
        try:
            errors.append(float(parts[-1]))
        except:
            pass
    return 0 if len(errors) == 0 else max(errors)


class FeedbackEstimatorState:
    def __init__(self):
        self.previous_state = None
        self.error_prop = None
        self.perf_est = None
        self.n_bits = 32
        self.max_merge_dist = 2
        self.max_err = 0.01
        self.stop = False

    def allowed_n_bits(self):
        return [16, 32, 64]

    def allowed_max_merge_dist(self):
        return [4, 3, 2, 1, 0]

    def num_options(self):
        return len(self.allowed_n_bits()) * len(self.allowed_max_merge_dist())

    def cur_option_num(self):
        res = self.allowed_n_bits().index(self.n_bits)
        res *= len(self.allowed_max_merge_dist())
        res += self.allowed_max_merge_dist().index(self.max_merge_dist)
        return res

    def set_option_from_num(self, opt):
        self.max_merge_dist = self.allowed_max_merge_dist()[opt % len(self.allowed_max_merge_dist())]
        self.n_bits = self.allowed_n_bits()[opt // len(self.allowed_max_merge_dist())]

    def tried_combinations(self, append=None):
        if append is None:
            append = []
        append += [(self.n_bits, self.max_merge_dist)]
        if self.previous_state:
            return self.previous_state.tried_combinations(append)
        return list(set(append))

    def set_next_untried_combination_with_direction(self, direction=0):
        combinations = set(self.tried_combinations())
        opt = self.cur_option_num()

        def fosc():
            yield 0
            n = 1
            while True:
                yield n
                yield -n
                n += 1

        variation = fosc()
        while (self.n_bits, self.max_merge_dist) in combinations:
            tmpdir = direction + next(variation)
            thisopt = opt + tmpdir
            if thisopt >= 0 and thisopt < self.num_options():
                self.set_option_from_num(opt + tmpdir)

    def advance_state(self, pe, ep):
        self.error_prop = ep
        self.perf_est = pe
        self.previous_state = copy(self)
        self.error_prop = None
        self.perf_est = None

        if pe > 0 and ep <= self.max_err:
            self.stop = True
            vprint('stopping; conditions fulfilled', file=sys.stderr)
            return

        num_options = len(self.allowed_n_bits()) * len(self.allowed_max_merge_dist())
        if len(self.tried_combinations()) == num_options:
            self.stop = True
            vprint('stopping; search space exhausted', file=sys.stderr)
            return

        option = self.cur_option_num()
        # option = 0                 : max speed
        # option = self.num_options(): min error
        direction = 0
        if ep > self.max_err and pe == 1:
            direction = +1
        elif ep > self.max_err and pe == 0:
            direction = -1
        elif ep > self.max_err and pe == -1:
            direction = -1
        elif ep <= self.max_err and pe == 1:
            pass
        elif ep <= self.max_err and pe == 0:
            direction = -1
        elif ep <= self.max_err and pe == -1:
            direction = -len(self.allowed_n_bits())

        if direction == 0:
            self.stop = True
            vprint('stopping; direction = 0', file=sys.stderr)
            return
        vprint('moving with direction', direction, file=sys.stderr)
        self.set_next_untried_combination_with_direction(direction)


    def dta_args(self):
        if self.stop:
            return 'STOP'
        return '--totalbits=' + str(self.n_bits) + ' --similarbits=' + str(self.max_merge_dist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TAFFO Feedback Estimator Experimental Tool. Outputs on stdout the command line parameters to DTA for the next compilation or STOP if the compilation loop should be interrupted.')
    parser.add_argument('--pe-out', '-p', type=str, help='file containing the output of the performance estimator')
    parser.add_argument('--ep-out', '-e', type=str, help='file containing the output of the error propagator')
    parser.add_argument('--pe-val', type=int)
    parser.add_argument('--ep-val', type=float)
    parser.add_argument('--init', '-i', action='store_true', help='initialize the state for a new compilation')
    parser.add_argument('--max-err', '-E', type=float, help='error threshold', default=0.01)
    parser.add_argument('--state', '-s', type=str, help='state file which will carry over across multiple compilations', default='fe-state.bin')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    if args.verbose:
        vprint = print

    if args.init:
        state = FeedbackEstimatorState()
    else:
        pe = args.pe_val if args.pe_val is not None else parse_perfest(Path(args.pe_out))
        ep = args.ep_val if args.ep_val is not None else parse_errorprop(Path(args.ep_out))
        with open(args.state, 'rb') as f:
            state = pickle.load(f)
        state.advance_state(pe, ep)

    with open(args.state, 'wb') as f:
        pickle.dump(state, f)
    print(state.dta_args())
