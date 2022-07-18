; RUN: opt -load %errorproplib --load-pass-plugin=%errorproplib --passes="taffoerr" -debug-only=errorprop -S %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: Possible overflow detected for instruction mul.
; CHECK: Possible overflow detected for instruction mul1.
; CHECK: Possible overflow detected for instruction add.
; CHECK: Possible overflow detected for instruction .

; CHECK: %mul = mul nsw i32 %a, %b, !taffo.info !7, !taffo.abserror !9
; CHECK: %mul1 = mul nsw i32 %a, %b, !taffo.info !7, !taffo.abserror !9
; CHECK: %add = add nsw i32 %mul, %mul1, !taffo.info !10, !taffo.abserror !13
; CHECK: ret i32 %add, !taffo.info !10, !taffo.abserror !13

; Function Attrs: noinline nounwind uwtable
define i32 @foo(i32 %a, i32 %b) #0 !taffo.funinfo !0 {
entry:
  %mul = mul nsw i32 %a, %b, !taffo.info !7
  %mul1 = mul nsw i32 %a, %b, !taffo.info !7
  %add = add nsw i32 %mul, %mul1, !taffo.info !9
  ret i32 %add, !taffo.info !9
}

!0 = !{i32 1, !1, i32 1, !5}
!1 = !{!2, !3, !4}
!2 = !{!"fixp", i32 -32, i32 16}
!3 = !{double -1.000000e+02, double 2.000000e+02}
!4 = !{double 1.000000e-05}
!5 = !{!2, !6, !4}
!6 = !{double -5.000000e+04, double 1.000000e+04}
!7 = !{!2, !8, i1 false}
!8 = !{double -5.000000e+06, double 2.000000e+06}
!9 = !{!10, !11, i1 false}
!10 = !{!"fixp", i32 32, i32 16}
!11 = !{double -1.000000e+07, double 4.000000e+06}

; CHECK: !9 = !{double 0x3FE010624DE0B01A}
; CHECK: !13 = !{double 0x3FF010624DE0B01A}
