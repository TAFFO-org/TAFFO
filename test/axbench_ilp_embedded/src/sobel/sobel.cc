#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include "picture_data.hpp"
#include "benchmark.hpp"

#define IMG_W 80
#define IMG_H 80
#define RAW_PRINT


static const float __attribute((annotate("scalar()"))) kx[][3] =
                {
                        { -1, -2, -1 },
                        {  0,  0,  0 },
                        {  1,  2,  1 }
                };

static const float __attribute((annotate("scalar()"))) ky[][3] =
                {
                        { -1, 0, 1 },
                        { -2, 0, 2 },
                        { -1, 0, 1 }
                };
		
#define WINDOW(image, x, y, window) {  \
        window[0][0] = image[ y - 1 ][ x - 1 ];   \
        window[0][1] = image[ y - 1 ][ x ];       \
        window[0][2] = image[ y - 1 ][ x + 1 ];   \
        window[1][0] = image[ y ][ x - 1 ];       \
        window[1][1] = image[ y ][ x ];           \
        window[1][2] = image[ y ][ x + 1 ];       \
        window[2][0] = image[ y + 1 ][ x - 1 ];   \
        window[2][1] = image[ y + 1 ][ x ];       \
        window[2][2] = image[ y + 1 ][ x + 1 ];   \
}

#define HALF_WINDOW(image, x, y, window) { \
        window[0][0] = (x == 0 || y == 0                 ) ? 0 : image[y - 1][x - 1]; \
        window[0][1] = (y == 0                           ) ? 0 : image[y - 1][x];     \
        window[0][2] = (x == IMG_W - 1 || y == 0         ) ? 0 : image[y - 1][x + 1]; \
        window[1][0] = (x == 0                           ) ? 0 : image[y][x - 1];     \
        window[1][1] =                                           image[y][x];         \
        window[1][2] = (x == IMG_W - 1                   ) ? 0 : image[y][x + 1];     \
        window[2][0] = (x == 0 || y == IMG_H - 1         ) ? 0 : image[y + 1][x - 1]; \
        window[2][1] = (y == IMG_H - 1                   ) ? 0 : image[y + 1][x];     \
        window[2][2] = (x == IMG_W - 1 || y == IMG_H - 1 ) ? 0 : image[y + 1][x + 1]; \
}


static float convolve(const float w[][3] __attribute((annotate("scalar()"))),
               const float k[][3] __attribute((annotate("scalar()"))))
{
  float __attribute((annotate("scalar(range(-2,2) final)"))) r ;
  float __attribute((annotate("scalar()"))) rr ;
  r = 0.0 ;
  for( int j = 0 ; j < 3 ; j++ ) {
    for ( int i = 0 ; i < 3 ; i++ ) {
      rr = w[i][j] * k[j][i] ;
      r +=  rr;
    }
  }
  return r;
}


static float sobel(const float w[][3] __attribute((annotate("scalar()"))))
{
  float __attribute((annotate("scalar()"))) sx;
  float __attribute((annotate("scalar()"))) sy;
  float __attribute((annotate("scalar()"))) s;
  float __attribute((annotate("scalar(range(1e-1, 8) final)"))) ss;

  sx = convolve(w, ky);
  sy = convolve(w, kx);

  ss = sx * sx + sy * sy;
  s = sqrtf(ss);
  if (s >= (256 / sqrtf(256 * 256 + 256 * 256)))
    s = 255 / sqrtf(256 * 256 + 256 * 256);

  return s ;
}


#ifndef BENCH_MAIN
#define BENCH_MAIN main
#endif
extern "C" int BENCH_MAIN(int argc, char *argv[])
{
  float __attribute((annotate("scalar(range(0,1) final)"))) image[IMG_H][IMG_W];
  float __attribute((annotate("target('out') scalar()"))) output[IMG_H][IMG_W];
  float __attribute((annotate("target('s') scalar()"))) w[3][3] = {0};
  int x, y, i;
  
  i = 0;
  for (y = 0; y < IMG_H; y++) {
    for (x = 0; x < IMG_W; x++) {
      image[y][x] = pic_01.pix[i++] * 0.30 / 256.0;
      image[y][x] += pic_01.pix[i++] * 0.59 / 256.0;
      image[y][x] += pic_01.pix[i++] * 0.11 / 256.0;
    }
  }
  
  uint64_t kernel_time = 0;
  AxBenchTimer timer;

  y = 0;
  for (x = 0 ; x < IMG_W; x++) {
    HALF_WINDOW(image, x, y, w);
    output[y][x] = sobel(w);
  }

  for (y = 1 ; y < (IMG_H - 1) ; y++) {
    x = 0;
    HALF_WINDOW(image, x, y, w);
    output[y][x] = sobel(w);

    for (x = 1; x < IMG_W - 1; x++) {
      WINDOW(image, x, y, w) ;
      output[y][x] = sobel(w);
    }

    x = IMG_W - 1;
    HALF_WINDOW(image, x, y, w);
    output[y][x] = sobel(w);
  }

  y = IMG_H - 1;
  for (x = 0 ; x < IMG_W; x++) {
    HALF_WINDOW(image, x, y, w);
    output[y][x] = sobel(w);
  }
  
  kernel_time = timer.cyclesSinceReset();
  printf("kernel time = %lld cycles\n", kernel_time);
  
  /* print */
#ifdef RAW_PRINT
  for (y = 0; y < IMG_H; y++) {
    for (x = 0; x < IMG_W; x++) {
      printf("%f, ", output[y][x]);
    }
    printf("\n");
  }
#else
  printf("P3\n");
  printf("%d %d\n", IMG_W, IMG_H);
  printf("256\n");
  for (y = 0; y < IMG_H; y++) {
    for (x = 0; x < IMG_W; x++) {
      printf("%d ", (int)(output[y][x] * sqrtf(256 * 256 + 256 * 256) + 0.5));
      printf("%d ", (int)(output[y][x] * sqrtf(256 * 256 + 256 * 256) + 0.5));
      printf("%d ", (int)(output[y][x] * sqrtf(256 * 256 + 256 * 256) + 0.5));
    }
    printf("\n");
  }
#endif

  return 0;
}
