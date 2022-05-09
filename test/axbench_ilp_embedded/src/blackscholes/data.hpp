#pragma once

//Precision to use for calculations
#define fptype float

struct blackscholes_line {
  fptype s, stk, r, divq, v, t;
  char OptionType;
  fptype divs, DGrefval;
};

#define BLACKSCHOLES_DATA_SIZE 2000
extern const blackscholes_line blackscholes_data[BLACKSCHOLES_DATA_SIZE];

