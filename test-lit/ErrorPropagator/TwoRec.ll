; RUN: opt -load %errorproplib --load-pass-plugin=%errorproplib --passes="taffoerr" -recur 4 -S %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: %add = add nsw i32 %x, %y, !taffo.info !5, !taffo.abserror !7
; CHECK: %call = call i32 @bar(i32 %add), !taffo.info !8, !taffo.abserror !10
; CHECK: ret i32 %call, !taffo.abserror !10

define i32 @foo(i32 %x, i32 %y) !taffo.funinfo !0 {
entry:
  %add = add nsw i32 %x, %y, !taffo.info !5
  %call = call i32 @bar(i32 %add), !taffo.info !7
  ret i32 %call
}

; CHECK: %mul = mul nsw i32 %x, %x, !taffo.info !8, !taffo.abserror !13
; CHECK: %call = call i32 @foo(i32 %mul, i32 %x), !taffo.info !14, !taffo.abserror !16
; CHECK: %retval.0 = phi i32 [ %call, %if.then ], [ %x, %if.else ], !taffo.info !8, !taffo.abserror !16
; CHECK: ret i32 %retval.0, !taffo.abserror !16

define i32 @bar(i32 %x) !taffo.funinfo !9 {
entry:
  %cmp = icmp slt i32 %x, 10000
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %mul = mul nsw i32 %x, %x, !taffo.info !7
  %call = call i32 @foo(i32 %mul, i32 %x), !taffo.info !11
  br label %return

if.else:                                          ; preds = %entry
  br label %return

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ %call, %if.then ], [ %x, %if.else ], !taffo.info !7
  ret i32 %retval.0
}

!0 = !{i32 1, !1, i32 1, !1}
!1 = !{!2, !3, !4}
!2 = !{!"fixp", i32 32, i32 5}
!3 = !{double 3.125000e-02, double 3.125000e+02}
!4 = !{double 1.250000e-02}
!5 = !{!2, !6, i1 0}
!6 = !{double 6.250000e-02, double 6.250000e+02}
!7 = !{!2, !8, i1 0}
!8 = !{double 1.250000e-01, double 1.250000e+07}
!9 = !{i32 1, !10}
!10 = !{!2, !6, !4}
!11 = !{!2, !12, i1 0}
!12 = !{double 2.500000e-01, double 2.500000e+07}

; CHECK: !7 = !{double 2.500000e-02}
; CHECK: !10 = !{double 0x41D8B6AB8B85A04A}
; CHECK: !13 = !{double 0x402F40147AE147AF}
; CHECK: !16 = !{double 0x41B8DC877BD0D827}
