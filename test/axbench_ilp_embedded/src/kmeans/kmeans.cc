#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include "picture_data.hpp"
#include "benchmark.hpp"

#define NUM_CLUSTERS 12
#define IMG_W 80
#define IMG_H 80
#define RAW_PRINT


static float euclideanDistance(float __attribute__((annotate("scalar()"))) rp, float __attribute__((annotate("scalar()"))) gp, float __attribute__((annotate("scalar()"))) bp, float __attribute__((annotate("scalar()"))) rc, float __attribute__((annotate("scalar()"))) gc, float __attribute__((annotate("scalar()"))) bc)
{
  float __attribute__((annotate("scalar()"))) rd;
  float __attribute__((annotate("scalar()"))) gd;
  float __attribute__((annotate("scalar()"))) bd;
  rd = rp - rc;
  gd = gp - gc;
  bd = bp - bc;
  return sqrtf(rd*rd + gd*gd + bd*bd);
}


#ifndef BENCH_MAIN
#define BENCH_MAIN main
#endif
extern "C" int BENCH_MAIN(int argc, char *argv[])
{
  float __attribute__((annotate("target('img') scalar(range(0,1))"))) img[IMG_W][IMG_H][3];
  int img_cluster[IMG_W][IMG_H];
  float __attribute__((annotate("target('clusters') scalar(range(1,3000) final)"))) clusters[NUM_CLUSTERS][3];
  int __attribute__((annotate("scalar(disabled range(1,3000) final)"))) clusters_n[NUM_CLUSTERS];
  int i, j;
  int x, y;
  int c;
  float __attribute__((annotate("scalar(range(0,1) final)"))) t;
  float __attribute__((annotate("scalar()"))) distance, __attribute__((annotate("scalar()"))) minDistance;
  
  i = 0;
  for (y = 0; y < IMG_H; y++) {
    for (x = 0; x < IMG_W; x++) {
      img[y][x][0] = pic_01.pix[i++] / 256.0;
      img[y][x][1] = pic_01.pix[i++] / 256.0;
      img[y][x][2] = pic_01.pix[i++] / 256.0;
    }
  }
  
  uint64_t kernel_time = 0;
  AxBenchTimer timer;

  /* init */
  for (i = 0; i < NUM_CLUSTERS; i++) {
    t = (float)rand() / RAND_MAX;
    clusters[i][0] = t;
    t = (float)rand() / RAND_MAX;
    clusters[i][1] = t;
    t = (float)rand() / RAND_MAX;
    clusters[i][2] = t;
  }
  
  /* assign */
  for (y = 0; y < IMG_H; y++) {
    for (x = 0; x < IMG_W; x++) {
      minDistance = euclideanDistance(img[y][x][0], img[y][x][1], img[y][x][2], clusters[0][0], clusters[0][1], clusters[0][2]);
      c = 0;
      for (i = 1; i < NUM_CLUSTERS; i++) {
        distance = euclideanDistance(img[y][x][0], img[y][x][1], img[y][x][2], clusters[i][0], clusters[i][1], clusters[i][2]);
        if (distance < minDistance) {
          minDistance = distance;
          c = i;
        }
      }
      img_cluster[y][x] = c;
    }
  }

  /* Recenter */
  for (c  = 0; c < NUM_CLUSTERS; ++c) {
    clusters[c][0] = 0.;
    clusters[c][1] = 0.;
    clusters[c][2] = 0.;
    clusters_n[c] = 0;
  }
  for (y = 0; y < IMG_H; y++) {
    for (x = 0; x < IMG_W; x++) {
      clusters[img_cluster[y][x]][0] += img[y][x][0];
      clusters[img_cluster[y][x]][1] += img[y][x][1];
      clusters[img_cluster[y][x]][2] += img[y][x][2];
      clusters_n[img_cluster[y][x]] += 1;
    }
  }

  for (c  = 0; c < NUM_CLUSTERS; ++c) {
    if (clusters_n[c] != 0) {
      clusters[c][0] /= clusters_n[c];
      clusters[c][1] /= clusters_n[c];
      clusters[c][2] /= clusters_n[c];
    }
  }

  for (y = 0; y < IMG_H; y++) {
    for (x = 0; x < IMG_W; x++) {
      img[y][x][0] = clusters[img_cluster[y][x]][0];
      img[y][x][1] = clusters[img_cluster[y][x]][1];
      img[y][x][2] = clusters[img_cluster[y][x]][2];
    }
  }
  
  kernel_time = timer.cyclesSinceReset();
  printf("kernel time = %lld cycles\n", kernel_time);
  
  /* print */
#ifdef RAW_PRINT
  for (y = 0; y < IMG_H; y++) {
    for (x = 0; x < IMG_W; x++) {
      printf("%f, ", img[y][x][0]);
      printf("%f, ", img[y][x][1]);
      printf("%f, ", img[y][x][2]);
    }
    printf("\n");
  }
#else
  printf("P3\n");
  printf("%d %d\n", IMG_W, IMG_H);
  printf("256\n");
  for (y = 0; y < IMG_H; y++) {
    for (x = 0; x < IMG_W; x++) {
      printf("%d ", (int)(img[y][x][0] * 256));
      printf("%d ", (int)(img[y][x][1] * 256));
      printf("%d ", (int)(img[y][x][2] * 256));
    }
    printf("\n");
  }
#endif
  
  return 0;
}
