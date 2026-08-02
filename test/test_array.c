#include <stdio.h>

int main() {
    // Array con annotazioni di range per abilitare Taffo
    float __attribute__((annotate("scalar(range(1, 10))"))) array_test[3];
    
    // Assegnazioni manuali che VRA deve intercettare tramite extractGEPOffset
    array_test[0] = 5.0f;
    array_test[1] = 2.0f;
    array_test[2] = 8.0f;

    // Utilizziamo le variabili per non farle scartare dal compilatore (Dead Code Elimination)
    float result = array_test[0] + array_test[1] + array_test[2];
    
    printf("Result: %f\n", result);
    return 0;
}