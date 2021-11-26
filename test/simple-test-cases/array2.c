///TAFFO_TEST_ARGS -Xvra -propagate-all -Xvra -max-unroll=10
#include <stdio.h>

/* This dumb test highlights the performance speed up that could be obtained with
   the use of optimization pass. */

#define N 100000

int main()
{
  __attribute__((annotate("scalar(range(-3000, 3000) final)"))) float v[N];

  for (int i=0;i<N;i++)
    v[i]= (i%(N/30)) ? 0 : 1;

  v[0]=1.1;
  v[1]=1.1;
  for (int i=2;i<N;i++) {
    for (int j=0;j<N;j++) {
       if (j==i)
         continue;
       v[i]+=(v[j]/10);
       if (v[i]>100)
         v[i]=0.1;
    }

  }
  printf ("ris %f \n",v[N-1]);
  return 0;
}
