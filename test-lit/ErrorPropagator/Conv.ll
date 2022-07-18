; RUN: opt -load %errorproplib --load-pass-plugin=%errorproplib --passes="taffoerr" -S %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [17 x i8] c"hello world, %d\0A\00", align 1


; Function Attrs: noinline nounwind optnone uwtable
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i64, align 8
  %d = alloca i64, align 8
  store i32 0, i32* %retval, align 4
  store i32 10, i32* %a, align 4, !taffo.info !2
  store i32 11, i32* %b, align 4, !taffo.info !6
  %0 = load i32, i32* %a, align 4
  %1 = load i32, i32* %b, align 4
  %add = add nsw i32 %0, %1, !taffo.info !9
  store i32 %add, i32* %a, align 4
  %2 = load i32, i32* %a, align 4
  %conv = sext i32 %2 to i64, !taffo.info !11
  store i64 %conv, i64* %c, align 8
  %3 = load i32, i32* %b, align 4
  %conv1 = sext i32 %3 to i64, !taffo.info !13
  store i64 %conv1, i64* %d, align 8
  %4 = load i64, i64* %c, align 8
  %5 = load i64, i64* %d, align 8
  %sub = sub nsw i64 %4, %5, !taffo.info !14
  %conv2 = trunc i64 %sub to i32, !taffo.info !16
  store i32 %conv2, i32* %b, align 4
  %6 = load i32, i32* %b, align 4
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([17 x i8], [17 x i8]* @.str, i32 0, i32 0), i32 %6)
  ret i32 0
}

declare i32 @printf(i8*, ...) #1

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.1 (https://git.llvm.org/git/clang.git/ 0e746072ed897a85b4f533ab050b9f506941a097) (git@github.com:llvm-mirror/llvm.git c36325f946ec29943d593d6b33664cb564df99ea)"}

!2 = !{!3, !4, !5}
!3 = !{!"fixp", i32 32, i32 5}
!4 = !{double 2.500000e-01, double 3.750000e-01}
!5 = !{double 1.000000e-03}
!6 = !{!3, !7, !8}
!7 = !{double 3.125000e-01, double 4.375000e-01}
!8 = !{double 2.000000e-03}
!9 = !{!3, !10, i1 0}
!10 = !{double 5.625000e-01, double 8.125000e-01}
!11 = !{!12, !10, i1 0}
!12 = !{!"fixp", i32 64, i32 5}
!13 = !{!12, !7, i1 0}
!14 = !{!12, !15, i1 0}
!15 = !{double 1.250000e-01, double 5.000000e-01}
!16 = !{!3, !15, i1 0}

; CHECK-DAG: !{double 0.000000e+00}
; CHECK-DAG: !{double 6.250000e-02}
; CHECK-DAG: !{double 3.125000e-02}

;  store i32 0, i32* %retval, align 4, !taffo.abserror !2
;  store i32 10, i32* %a, align 4, !taffo.info !5, !taffo.abserror !4
;  store i32 11, i32* %b, align 4, !taffo.info !9, !taffo.abserror !4
;  %0 = load i32, i32* %a, align 4, !taffo.abserror !4
;  %1 = load i32, i32* %b, align 4, !taffo.abserror !4
;  %add = add nsw i32 %0, %1, !taffo.info !12, !taffo.abserror !3
;  store i32 %add, i32* %a, align 4, !taffo.abserror !3
;  %2 = load i32, i32* %a, align 4, !taffo.abserror !3
;  %conv = sext i32 %2 to i64, !taffo.info !14, !taffo.abserror !3
;  store i64 %conv, i64* %c, align 8, !taffo.abserror !3
;  %3 = load i32, i32* %b, align 4, !taffo.abserror !4
;  %conv1 = sext i32 %3 to i64, !taffo.info !16, !taffo.abserror !4
;  store i64 %conv1, i64* %d, align 8, !taffo.abserror !4
;  %4 = load i64, i64* %c, align 8, !taffo.abserror !3
;  %5 = load i64, i64* %d, align 8, !taffo.abserror !4
;  %sub = sub nsw i64 %4, %5, !taffo.info !17, !taffo.abserror !4
;  %conv2 = trunc i64 %sub to i32, !taffo.info !19, !taffo.abserror !4
;  store i32 %conv2, i32* %b, align 4, !taffo.abserror !4
;  %6 = load i32, i32* %b, align 4, !taffo.abserror !4
;  ret i32 0, !taffo.abserror !2
;  !2 = !{double 0.000000e+00}
;  !3 = !{double 6.250000e-02}
;  !4 = !{double 3.125000e-02}


