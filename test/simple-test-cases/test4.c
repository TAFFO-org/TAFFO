///TAFFO_TEST_ARGS -Xvra -propagate-all


/* compile with -O1 */


float random(void)
{
  static unsigned int seed = 123456;
  seed = (seed * 0xc8a1248f + 42) % 0xfffffffe;
  return (double)seed / (double)(0xffffffff);
}


float test(int p1, int p2, int p3)
{
  float f1 __attribute((annotate("scalar(range(-32767, 32767)"))) = random(); 
  float f2 __attribute((annotate("scalar(range(-32767, 32767)"))) = random();
  float phi __attribute((annotate("scalar(range(-32767, 32767)")));
  
  if (p1)
    phi = 1.0;
  else
    phi = 1.2;
  phi *= p3 ? f1 : f2;
  if (p2)
    phi += 0.8;
  else
    phi += 0.3;
  return phi;
}
  
