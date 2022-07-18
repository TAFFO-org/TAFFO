; RUN: opt -load %errorproplib --load-pass-plugin=%errorproplib --passes="taffoerr" -S %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = global i32 5, align 4, !taffo.info !0
@b = global i32 10, align 4, !taffo.info !4

; CHECK: %0 = load i32, i32* @a, align 4, !taffo.abserror !3
; CHECK: %add = add nsw i32 %c, %0, !taffo.info !10, !taffo.abserror !6
; CHECK: store i32 %add, i32* @a, align 4, !taffo.abserror !6
; CHECK: %1 = load i32, i32* @b, align 4, !taffo.abserror !6
; CHECK: %mul = mul nsw i32 %c, %1, !taffo.info !12, !taffo.abserror !14
; CHECK: store i32 %mul, i32* @b, align 4, !taffo.abserror !14
; CHECK: %2 = load i32, i32* @a, align 4, !taffo.abserror !6
; CHECK: %3 = load i32, i32* @b, align 4, !taffo.abserror !14
; CHECK: %add1 = add nsw i32 %2, %3, !taffo.info !15, !taffo.abserror !17
; CHECK: ret i32 %add1, !taffo.abserror !17

; Function Attrs: noinline nounwind uwtable
define i32 @foo(i32 %c) !taffo.funinfo !7 {
entry:
  %0 = load i32, i32* @a, align 4
  %add = add nsw i32 %c, %0, !taffo.info !11
  store i32 %add, i32* @a, align 4
  %1 = load i32, i32* @b, align 4
  %mul = mul nsw i32 %c, %1, !taffo.info !13
  store i32 %mul, i32* @b, align 4
  %2 = load i32, i32* @a, align 4
  %3 = load i32, i32* @b, align 4
  %add1 = add nsw i32 %2, %3, !taffo.info !15
  ret i32 %add1
}

!0 = !{!1, !2, !3}
!1 = !{!"fixp", i32 32, i32 5}
!2 = !{double 4.000000e+00, double 6.000000e+00}
!3 = !{double 1.000000e-02}
!4 = !{!1, !5, !6}
!5 = !{double 9.000000e+00, double 1.100000e+01}
!6 = !{double 2.000000e-02}
!7 = !{i32 1, !8}
!8 = !{!1, !9, !10}
!9 = !{double 2.000000e+00, double 3.000000e+00}
!10 = !{double 1.000000e-02}
!11 = !{!1, !12, i1 0}
!12 = !{double 6.000000e+00, double 9.000000e+00}
!13 = !{!1, !14, i1 0}
!14 = !{double 5.760000e+02, double 1.056000e+03}
!15 = !{!1, !16, i1 0}
!16 = !{double 5.820000e+02, double 1.065000e+03}

; CHECK: !14 = !{double 1.702000e-01}
; CHECK: !17 = !{double 1.902000e-01}
