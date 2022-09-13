Benchmark applications:
- limited subset of PolyBench/C 4.2
- ones we used for JSA already available on Enterprise, with few modifications:
  - Taffo annotations to implement fixed-point computations
  - Modifications to make them work on our CPU
  - In particular:
    * tolti parametri da funzioni init_array e kernel_<bench>, vengono usate variabili globali
    * i risultati dei bench vengono copiati (e convertiti, se fixed) in
      variabili <nomeVettore/Matrice>_float
    * annotazione per le conversioni a fixed (nessuna annotazione per float)
    * annotazione applicata indiscriminatamente a TUTTE le variabili float
    * vengono usati gli input di default dei PolyBench
    * input dataset ha dimensioni ridotte rispetto a minime supportate da PolyBench
    * fixato indicizzazione di vettori con indici costanti in durbin
    * le costanti di tipo floating-point (presenti nei bench durbin, gemver, gramschmidt,
      ludcmp e nella funzione comune sqrtf)
      sono state trasformate nelle corrispondenti variabili floating-point, opportunamente
      annotate secondo la sintassi di Taffo
    * funzione sqrtf rinominata in sqrtf_PB, in modo che il compilatore non inferisca
      l'istruzione fsqrt.s dell'estensione F dell'ISA RISC-V

## How to compile 

In mixed mode:
```shell
mixedmode=1 ./compile.sh
```

In fixed mode:
```shell
./compile.sh
```
