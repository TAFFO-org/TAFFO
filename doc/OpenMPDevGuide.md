# OpenMP Dev Guide
This document provides technical details related to the OpenMP integration. In particular, it describes:
- An introduction to the OpenMP runtime library functions and how to patch them
- The approach used and the modifications to the TAFFO passes
- The possible next steps to enhance the current integration

Check out [the OpenMP user guide](./OpenMPUserGuide.md) to learn how to use TAFFO with OpenMP.

## Patching OpenMP Runtime
The OpenMP implementation provided by the LLVM Framework inserts various indirect functions to manage parallelism;
these functions are OpenMP intrinsics, which are not defined in the LLVM-IR module that is being compiled.
Therefore, TAFFO cannot natively follow the flow of execution.
 
The source below describes a simple OpenMP example program:
```c++
#include <stdio.h>

int main() {
  float container[16] __attribute__((annotate("target('container') scalar()")));
  // container is a shared variable between the OpenMP threads
  #pragma omp parallel for shared(container)
  for (int i = 0; i < 16; i++) {
    // Do parallel computation
    float result = i*0.05;
    
    // Save the result in the shared variable
    container[i] = result;
  }

  return 0;
}
```

A simplified `main` in the LLVM IR produced by TAFFO is as follows:

```llvm
define dso_local i32 @main() #0 {
entry:
  // Setup variables
  // Setup OpenMP threads with OpenMP runtime library functins
  call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* %.kmpc_loc.addr, i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, float*)* @.omp_outlined. to void (i32*, i32*, ...)*), float* %container)
  ret i32 0
}
```

The ```main``` contains only library functions and variables allocation.
The logic contained in the parallel region is outlined in another function, the ```.omp_outlined.```.
The function directly called is the ```__kmpc_fork_call```, whose source is not present in the LLVM IR.
Additionally, the ```__kmpc_fork_call``` receives the following arguments:
- Two parameters for internal OpenMP configurations.
- The reference to the ```.omp_outlined.```, where the actual logic is.
- A dynamic number of variables that corresponds to the reference to the shared variables.

### Taken approach
To bypass the runtime library functions, it is necessary to let TAFFO understand what functions will be called.

Although the integration started with the goal to instruct every pass to bypass the indirect function and analyze the outlined one,
the path was discarded since substantial modifications were required in every pass of TAFFO.
 
A simpler but more effective approach is currently implemented, with modifications only to the Initialization and Conversion passes.
Instead of resolving the indirections at each pass, the library functions are replaced with the outlined functions during the Initialization pass. 
Finally, during the last pass, the library function calls are re-inserted, accounting for the new fixed-point functions created by TAFFO.

The chosen approach revolves around so-called "trampolines". A trampoline is a custom function that directly calls an indirect function. 
This is convenient because it allows TAFFO to analyze the indirect calls as if they were direct calls, which are already supported.

#### Initialization pass
From the above example of the `@__kmpc_fork_call`, a corresponding `@__kmpc_fork_call_trampoline` is created and inserted in place of the original one. 
The body of this function is very simple and contains:
- the direct call to the outlined function, with the related arguments (the shared variables in OpenMP).
- an additional instruction that is used to preserve information about the original function. 
 In particular, the original function is saved in a local variable using a bitcast instruction. 
 This is necessary because otherwise the following optimization passes, such as Dead Code Elimination (DCE), 
 would remove the unused original function, making it impossible to reinsert the original calls in the Conversion pass.
 
Moreover, to take into account unsupported constructs, such as `omp reduce` and `omp tasks`, the Initializer pass
automatically disables the conversion for the shared variables in parallel regions where an unsupported construct is found.

#### Range Analysis and Data Type Allocation passes
TAFFO will regularly analyze the trampoline calls,  without any modification to these intermediate passes.

#### Conversion pass
At the end of the Conversion pass, it is necessary to transform back the trampolines into their original calls. 
To do so, the original function is picked from the saved bitcast and placed in place of the trampoline call, which will be deleted. 
The original function will have the relevant arguments replaced accordingly, specifically the indirect function to be called and the shared variables needed. 

In the above example, TAFFO will correctly convert the floating-point instructions in fixed-points, and
the call to `@__kmpc_fork_call` will contain `@.omp_outlined..2_fixp` instead of `@.omp_outlined.` and `i32* %foo.s6_26fixp` instead of `float* %foo`.

## Next Steps
The above sections describe where possible work can be done to reach a more complete integration of TAFFO and OpenMP.
### OpenMP reduce
The OpenMP reduction handling from Clang appears similar to the regular parallel code section handling.
A `@__kmpc_reduce` function (or `@__kmpc_reduce_nowait`) receives as arguments both the code to run and the data to work on. 

However, differently from the fork_call case, the data is passed as `void*`, which becomes `i8*` in LLVM IR. 
This becomes a problem because TAFFO is unable to apply any alias analysis on the reduction data, which is called `%.omp.reduction.red_list`. 

As a consequence, the code that corresponds to the reduction area maintains its usual data types, without any possible float to fixed-point conversion.

To avoid unexpected errors, conversion is disabled for shared variables, that always include the reduction variable.
The natural downside is blocking conversion also for other variables.

The problem could be tackled with: 
- targeted backtracking to find the usage of the reduction variable, which is a duplicated variable created from OpenMP and then inserted into the `%.omp.reduction.red_list`.
- enhancing the alias analysis to take into account the `%.omp.reduction.red_list` and the list of variables that will be passed to the reduce function(s).

### Loops inside parallel regions
The ValueRangeAnalysis pass is unable to correctly understand the range of trivial loops. A simple example could be as follows:
```c++
int main() {
    float foo __attribute((annotate("target('foo') scalar()"))) = 10;

    #pragma omp parallel for shared(foo)
    for (int i = 0; i < 5; i++) {
        #pragma omp critical
        foo += i;
    }
    printf("%f\n", foo);

    return 0;
}
```

The executable compiled with TAFFO prints 4.000000 instead of the expected 20.000000. 
The underlying problem is that the ScalarEvolution, used by the VRA to calculate the trip counts, fails to detect the range of the loop.

The problem could be tackled with additional simplification passes, but they do not seem effective and 
risk creating additional problems for the TAFFO's passes.

### OpenMP tasks
OpenMP tasks aren't supported since they cause errors in Conversion, specifically causing it to hit an assertion error 
(`assert(ip && "ip mandatory for non-instruction values");`).
In particular, the error is raised during the store of a float in an OpenMP struct, used for the shared variables in the task.

Although the behaviour has not been fixed, a workaround is present to manage them. 
To be conservative, TAFFO prevents the shared variables of parallel regions to be converted if they contain tasks. 

This could have undesired behaviour in cases such as this one:
```c++
int main() {
    float task_variable __attribute((annotate("target('task_variable') scalar()"))) = 1.0f;
    float not_task_variable __attribute((annotate("target('not_task_variable') scalar()"))) = 363;
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            {
                task_variable *= 2.0f;
            }
            #pragma omp task
            task_variable *= 3.0f;
            
            not_task_variable += 4;
        }
    }
}
```
The variable `not_task_variable` will not be converted even if it isn't used in the tasks, 
 since it is a shared variable.

The problem could be tackled in different ways:
- target specifically the variables used in the tasks (by analyzing the shared variables in the task OpenMP function).
 This could allow, in the example above, the conversion of `not_task_variable`, meanwhile disabling the conversion for `task_variable`.
- supporting OpenMP tasks, in a similar way to the current `__kmpc_fork_call` support. This will need a deeper investigation of
 the `assert(ip && "ip mandatory for non-instruction values");` error.

## Additional links
During the development of the integration, another team paved way for [parallelism-aware optimizations in LLVM](https://www.openmp.org/wp-content/uploads/OpenMPOpt-in-LLVM-SC20-JD.pdf).

This could be extremely interesting in solving the problems, in case the aforementioned indirect functions will be substituted by direct functions.
