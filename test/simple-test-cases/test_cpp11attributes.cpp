///TAFFO_TEST_ARGS -disable-vra

float oven [[clang::annotate("range -3000 3000")]] (int stuff [[clang::annotate("range -3000 3000")]], int baked, float cherry) 
{
  float cake [[clang::annotate("range -3000 3000")]] = baked + stuff;
  return cake + cherry;
}


