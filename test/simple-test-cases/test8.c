///TAFFO_TEST_ARGS -Xvra -propagate-all

typedef struct spell {
  float might;
  int mpcost;
} spell_t;


typedef struct chara {
  float hp;
  int mp;
  float mdef;
} chara_t;


float cast(spell_t *spell, chara_t *caster, chara_t *enemy)
{
  caster->mp -= spell->mpcost;
  __attribute((annotate("force_no_float range -32767 32767"))) float might = spell->might;
  __attribute((annotate("force_no_float range -32767 32767"))) float *ehp = &(enemy->hp);
  *ehp -= (might - enemy->mdef);
  return *ehp;
}


