// Copyright (c) 2007 Intel Corp.

// Black-Scholes
// Analytical method for calculating European Options
//
//
// Reference Source: Options, Futures, and Other Derivatives, 3rd Edition, Prentice
// Hall, John C. Hull,

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "benchmark.hpp"
#include "data.hpp"


//double max_otype, min_otype ;
//double max_sptprice, min_sptprice;
//double max_strike, min_strike;
//double max_rate, min_rate ;
//double max_volatility, min_volatility;
//double max_otime, min_otime ;
//double max_out_price, min_out_price;

#define DIVIDE 120.0

#define NUM_RUNS 1

typedef struct OptionData_ {
        fptype r;          // risk-free interest rate
        fptype divq;       // dividend rate
        fptype v;          // volatility
        fptype t;          // time to maturity or option expiration in years
                           //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)
        char OptionType;   // Option type.  "P"=PUT, "C"=CALL
        fptype divs;       // dividend vals (not used in this test)
        fptype DGrefval;   // DerivaGem Reference Value (unused)
} OptionData;

static fptype *prices;
static int numOptions;

static int    * otype;
static fptype __attribute((annotate("scalar(range(0.35,0.84) error(1e-8))"))) *sptprice;
static fptype __attribute((annotate("scalar(range(0.33,0.92) error(1e-8))"))) *strike;
static fptype __attribute((annotate("scalar(range(0.0275,0.1) error(0))"))) *rate;
static fptype __attribute((annotate("scalar(range(0.05,0.65) error(1e-8))"))) *volatility;
static fptype __attribute((annotate("scalar(range(0.05,1) error(0))"))) *otime;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
// Cumulative Normal Distribution Function
// See Hull, Section 11.8, P.243-244
#define inv_sqrt_2xPI 0.39894228040143270286

static fptype CNDF ( fptype __attribute((annotate("scalar()"))) InputX )
{
    int sign;

    fptype __attribute((annotate("scalar()"))) OutputX;
    fptype __attribute((annotate("scalar()"))) xInput;
    fptype __attribute((annotate("scalar()"))) xNPrimeofX;
    fptype __attribute((annotate("scalar(range(0,1) final)"))) expValues;
    fptype __attribute((annotate("scalar()"))) xK2;
    fptype __attribute((annotate("scalar()"))) xK2_2, xK2_3;
    fptype __attribute((annotate("scalar()"))) xK2_4, xK2_5;
    fptype __attribute((annotate("scalar()"))) xLocal, xLocal_1;
    fptype __attribute((annotate("scalar()"))) xLocal_2, xLocal_3;
    fptype __attribute((annotate("scalar(range(30,12412) final)"))) InputXX;

    // Check for negative value of InputX
    if (InputX < 0.0) {
        InputX = -InputX;
        sign = 1;
    } else
        sign = 0;

    xInput = InputX;
    InputXX = InputX;

    // Compute NPrimeX term common to both four & six decimal accuracy calcs
    expValues = exp(-0.5f * (InputXX * InputXX));
    xNPrimeofX = expValues;
    xNPrimeofX = xNPrimeofX * inv_sqrt_2xPI;

    xK2 = 0.2316419 * InputXX;
    xK2 = 1.0 + xK2;
    xK2 = 1.0 / xK2;
    xK2_2 = xK2 * xK2;
    xK2_3 = xK2_2 * xK2;
    xK2_4 = xK2_3 * xK2;
    xK2_5 = xK2_4 * xK2;

    xLocal_1 = xK2 * 0.319381530;
    xLocal_2 = xK2_2 * (-0.356563782);
    xLocal_3 = xK2_3 * 1.781477937;
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_4 * (-1.821255978);
    xLocal_2 = xLocal_2 + xLocal_3;
    xLocal_3 = xK2_5 * 1.330274429;
    xLocal_2 = xLocal_2 + xLocal_3;

    xLocal_1 = xLocal_2 + xLocal_1;
    xLocal   = xLocal_1 * xNPrimeofX;

    //printf("# xLocal: %10.10f\n", xLocal);

    xLocal   = 1.0 - xLocal;

    OutputX  = xLocal;

    //printf("# Output: %10.10f\n", OutputX);

    if (sign) {
        OutputX = 1.0 - OutputX;
    }

    return OutputX;
}

//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
static fptype BlkSchlsEqEuroNoDiv( fptype __attribute((annotate("scalar()"))) sptprice,
                            fptype __attribute((annotate("scalar()"))) strike,
			    fptype __attribute((annotate("scalar()"))) rate,
                            fptype __attribute((annotate("scalar()"))) volatility,
			    fptype __attribute((annotate("scalar()"))) time,
                            int otype,
			    float __attribute((annotate("scalar()"))) timet,
                            fptype* __attribute((annotate("scalar()"))) N1,
			    fptype* __attribute((annotate("scalar()"))) N2 )
{
    //printf("BlkSchlsEqEuroNoDiv %f %f %f %f %f %f\n", sptprice, strike, rate, volatility, time, timet);
    fptype __attribute((annotate("scalar()"))) OptionPrice;

    // local private working variables for the calculation
    //fptype xStockPrice;
    //fptype xStrikePrice;
    fptype __attribute((annotate("scalar()"))) xRiskFreeRate;
    fptype __attribute((annotate("scalar()"))) xVolatility;
    fptype __attribute((annotate("scalar()"))) xTime;
    fptype __attribute((annotate("scalar()"))) xSqrtTime;

    fptype __attribute((annotate("scalar()"))) logValues;
    fptype __attribute((annotate("scalar()"))) xLogTerm;
    fptype __attribute((annotate("scalar()"))) xD1;
    fptype __attribute((annotate("scalar()"))) xD2;
    fptype __attribute((annotate("scalar()"))) xPowerTerm;
    fptype __attribute((annotate("scalar()"))) xDen;
    fptype __attribute((annotate("scalar()"))) d1;
    fptype __attribute((annotate("scalar()"))) d2;
    fptype __attribute((annotate("scalar()"))) FutureValueX;
    fptype __attribute((annotate("scalar()"))) NofXd1;
    fptype __attribute((annotate("scalar()"))) NofXd2;
    fptype __attribute((annotate("scalar()"))) NegNofXd1;
    fptype __attribute((annotate("scalar()"))) NegNofXd2;

    //xStockPrice = sptprice;
    //xStrikePrice = strike;
    xRiskFreeRate = rate;
    xVolatility = volatility;
    xTime = time;


    xSqrtTime = sqrt(xTime);

    logValues = log( sptprice / strike );

    xLogTerm = logValues;


    xPowerTerm = xVolatility * xVolatility;
    xPowerTerm = xPowerTerm * 0.5;
    
    xD1 = xRiskFreeRate + xPowerTerm;
    xD1 = xD1 * xTime;
    xD1 = xD1 + xLogTerm;

    xDen = xVolatility * xSqrtTime;
    xD1 = xD1 / xDen;
    xD2 = xD1 -  xDen;

    d1 = xD1;
    d2 = xD2;

    NofXd1 = CNDF( d1 );

    if(NofXd1 > 1.0) {
        //std::cerr << "Greater than one!" << std::endl ;
    }
    //printf("# d1: %10.10f\n", NofXd1);

    NofXd2 = CNDF( d2 );
    if(NofXd2 > 1.0) {
         // std::cerr << "Greater than one!" << std::endl ;
    }
    //printf("# d2: %10.10f\n", NofXd2);

    *N1 = NofXd1 ;
    *N2 = NofXd2 ;

    FutureValueX = strike * ( exp( -(rate)*(time) ) );
    if (otype == 0) {
        OptionPrice = (sptprice * NofXd1) - (FutureValueX * NofXd2);

    } else {
        NegNofXd1 = (1.0 - NofXd1);
        NegNofXd2 = (1.0 - NofXd2);
        OptionPrice = (FutureValueX * NegNofXd2) - (sptprice * NegNofXd1);
    }

    return OptionPrice;
}


static double normalize(double in, double min, double max, double min_new, double max_new)
{
    return (((in - min) / (max - min)) * (max_new - min_new)) + min_new ;
}

static int bs_thread(void *tid_ptr) {
    int i, j;

    int tid = *(int *)tid_ptr;
    int start = tid * (numOptions);
    int end = start + (numOptions);
    fptype __attribute((annotate("target('price_orig') scalar()"))) price_orig;

    for (j=0; j<NUM_RUNS; j++) {
        for (i=start; i<end; i++) {
            /* Calling main function to calculate option value based on
             * Black & Scholes's equation.
             */
            fptype __attribute((annotate("scalar()"))) N1,
	      __attribute((annotate("scalar()"))) N2;
            float __attribute((annotate("scalar(range(0,1) error(0))"))) timet = 0;
/*
            double dataIn[6];
            double dataOut[1];

            dataIn[0]   = sptprice[i];
            dataIn[1]   = strike[i];
            dataIn[2]   = rate[i];
            dataIn[3]   = volatility[i];
            dataIn[4]   = otime[i];
            dataIn[5]   = otype[i];

#pragma parrot(input, "blackscholes", [6]dataIn)
*/
                price_orig = BlkSchlsEqEuroNoDiv( sptprice[i], strike[i],
                                         rate[i], volatility[i], otime[i],
                                         otype[i], timet, &N1, &N2);
/*
                dataOut[0] = price_orig;

#pragma parrot(output, "blackscholes", [1]<0.1; 0.9>dataOut)
*/
                prices[i] = price_orig;
        }
    }
    return 0;
}

#ifndef BENCH_MAIN
#define BENCH_MAIN main
#endif
extern "C" int BENCH_MAIN(int argc, char **argv)
{
    int i;
    int loopnum;
    fptype * buffer;
    int * buffer2;
    int rv;

    rv = numOptions = BLACKSCHOLES_DATA_SIZE;

    // alloc spaces for the option data
    prices = (fptype*)malloc(numOptions*sizeof(fptype));

#define PAD 256
#define LINESIZE 64

    buffer = (fptype *) malloc(5 * numOptions * sizeof(fptype) + PAD);
    sptprice = (fptype *) (((unsigned long long)buffer + PAD) & ~(LINESIZE - 1));
    strike = sptprice + numOptions;
    rate = strike + numOptions;
    volatility = rate + numOptions;
    otime = volatility + numOptions;

    buffer2 = (int *) malloc(numOptions * sizeof(fptype) + PAD);
    if (buffer2 == NULL) {
      printf("OUT OF MEMORY\n");
      return 0;
    }
    otype = (int *) (((unsigned long long)buffer2 + PAD) & ~(LINESIZE - 1));
    
    for (i=0; i<numOptions; i++) {
        otype[i]      = (blackscholes_data[i].OptionType == 'P') ? 1 : 0;
        sptprice[i]   = blackscholes_data[i].s / DIVIDE;
        strike[i]     = blackscholes_data[i].stk / DIVIDE;
        rate[i]       = blackscholes_data[i].r;
        volatility[i] = blackscholes_data[i].v;
        otime[i]      = blackscholes_data[i].t;
    }

    //serial version

    AxBenchTimer timer;

    int tid=0;
    bs_thread(&tid);

    uint32_t time = timer.cyclesSinceReset();
    printf("kernel time = %u units\n", time);

    //Write prices to output file
    for(i=0; i<numOptions; i++) {
      rv = printf("%.18f\n", prices[i]);
    }

    free(buffer);
    free(buffer2);
    free(prices);

    return 0;
}
