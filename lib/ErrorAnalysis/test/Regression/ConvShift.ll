; RUN: opt -load %errorproplib -errorprop -S %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [17 x i8] c"hello world, %d\0A\00", align 1


; Function Attrs: noinline nounwind optnone uwtable
define i32 @main() {
entry:
  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 0, i32* %retval, align 4
  store i32 10, i32* %a, align 4, !taffo.info !2
  store i32 11, i32* %b, align 4, !taffo.info !5
  %0 = load i32, i32* %a, align 4
  %1 = load i32, i32* %b, align 4
  %add = add nsw i32 %0, %1, !taffo.info !7
  store i32 %add, i32* %a, align 4
  %2 = load i32, i32* %a, align 4
  %shl = shl i32 %2, 3, !taffo.info !15
  store i32 %shl, i32* %a, align 4
  %3 = load i32, i32* %a, align 4
  %4 = load i32, i32* %b, align 4
  %sub = sub nsw i32 %3, %4, !taffo.info !9
  store i32 %sub, i32* %b, align 4
  %5 = load i32, i32* %b, align 4
  %shr = ashr i32 %5, 3, !taffo.info !18
  store i32 %shr, i32* %b, align 4
  %6 = load i32, i32* %a, align 4
  %7 = load i32, i32* %b, align 4
  %mul = mul nsw i32 %6, %7, !taffo.info !11
  store i32 %mul, i32* %a, align 4
  %8 = load i32, i32* %b, align 4
  %9 = load i32, i32* %a, align 4
  %div = sdiv i32 %8, %9, !taffo.info !13
  store i32 %div, i32* %b, align 4
  %10 = load i32, i32* %b, align 4
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str, i32 0, i32 0), i32 %10)
  ret i32 0
}

declare i32 @printf(i8*, ...)

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.1 (https://git.llvm.org/git/clang.git/ 0e746072ed897a85b4f533ab050b9f506941a097) (git@github.com:llvm-mirror/llvm.git c36325f946ec29943d593d6b33664cb564df99ea)"}
!2 = !{!3, !4, i1 false}
!3 = !{!"fixp", i32 32, i32 5}
!4 = !{double 0.250000e+00, double 0.375000e+00}
!5 = !{!3, !6, i1 false}
!6 = !{double 0.312500e+00, double 0.437500e+00}
!7 = !{!3, !8, i1 false}
!8 = !{double 0.562500e+00, double 0.812500e+00}
!9 = !{!3, !10, i1 false}
!10 = !{double 0.125000e+00, double 0.500000e+00}
!11 = !{!3, !12, i1 false}
!12 = !{double 2.250000e+00, double 1.300000e+01}
!13 = !{!3, !14, i1 false}
!14 = !{double 0.000000e+00, double 0.000000e+00}
!15 = !{!16, !17, i1 false}
!16 = !{!"fixp", i32 32, i32 8}
!17 = !{double 7.031250e-02, double 1.015625e-01}
!18 = !{!19, !20, i1 false}
!19 = !{!"fixp", i32 32, i32 2}
!20 = !{double 1.000000e+00, double 4.000000e+00}

; CHECK-DAG: !{double 3.125000e-02}
; CHECK-DAG: !{double 6.250000e-02}
; CHECK-DAG: !{double 2.812500e-01}
; CHECK-DAG: !{double 0x3FD2F40000000000}
; CHECK-DAG: !{double 0x3FE4039E06522C3F}

;  store i32 10, i32* %a, align 4, !taffo.info !3, !taffo.abserror !6
;  store i32 11, i32* %b, align 4, !taffo.info !7, !taffo.abserror !6
;  %0 = load i32, i32* %a, align 4, !taffo.abserror !6
;  %1 = load i32, i32* %b, align 4, !taffo.abserror !6
;  %add = add nsw i32 %0, %1, !taffo.info !9, !taffo.abserror !11
;  store i32 %add, i32* %a, align 4, !taffo.abserror !11
;  %2 = load i32, i32* %a, align 4, !taffo.abserror !11
;  %shl = shl i32 %2, 3, !taffo.info !12, !taffo.abserror !11
;  store i32 %shl, i32* %a, align 4, !taffo.abserror !11
;  %3 = load i32, i32* %a, align 4, !taffo.abserror !11
;  %4 = load i32, i32* %b, align 4, !taffo.abserror !6
;  %sub = sub nsw i32 %3, %4, !taffo.info !15, !taffo.abserror !6
;  store i32 %sub, i32* %b, align 4, !taffo.abserror !6
;  %5 = load i32, i32* %b, align 4, !taffo.abserror !6
;  %shr = ashr i32 %5, 3, !taffo.info !17, !taffo.abserror !20
;  store i32 %shr, i32* %b, align 4, !taffo.abserror !20
;  %6 = load i32, i32* %a, align 4, !taffo.abserror !11
;  %7 = load i32, i32* %b, align 4, !taffo.abserror !20
;  %mul = mul nsw i32 %6, %7, !taffo.info !21, !taffo.abserror !23
;  store i32 %mul, i32* %a, align 4, !taffo.abserror !23
;  %8 = load i32, i32* %b, align 4, !taffo.abserror !20
;  %9 = load i32, i32* %a, align 4, !taffo.abserror !23
;  %div = sdiv i32 %8, %9, !taffo.info !24, !taffo.abserror !26
;  store i32 %div, i32* %b, align 4, !taffo.abserror !26
;  %10 = load i32, i32* %b, align 4, !taffo.abserror !26
;  !6 = !{double 3.125000e-02}
;  !11 = !{double 6.250000e-02}
;  !20 = !{double 2.812500e-01}
;  !23 = !{double 0x3FD2F40000000000}
;  !26 = !{double 0x3FE4039E06522C3F}
