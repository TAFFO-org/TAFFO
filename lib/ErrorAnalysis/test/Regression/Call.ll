; RUN: opt -load %errorproplib -errorprop -S %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: %cmp = icmp sgt i32 %a, %b, !taffo.wrongcmptol !7
; CHECK: %sub = sub nsw i32 %a, %b, !taffo.info !8, !taffo.abserror !10
; CHECK: %sub1 = sub nsw i32 %b, %a, !taffo.info !8, !taffo.abserror !10
; CHECK: %retval.0 = phi i32 [ %sub, %if.then ], [ %sub1, %if.else ], !taffo.info !8, !taffo.abserror !10
; CHECK: ret i32 %retval.0, !taffo.abserror !10
; CHECK: %add = add nsw i32 %a, %b, !taffo.info !14, !taffo.abserror !10
; CHECK: %call = call i32 @bar(i32 %add, i32 %b), !taffo.info !8, !taffo.abserror !4
; CHECK: %add1 = add nsw i32 %add, %call, !taffo.info !15, !taffo.abserror !17
; CHECK: %sub = sub nsw i32 0, %call, !taffo.info !8, !taffo.abserror !18
; CHECK: %mul = mul nsw i32 %add, %sub, !taffo.info !19, !taffo.abserror !21
; CHECK: %retval.0 = phi i32 [ %add1, %if.then ], [ %mul, %if.end ], !taffo.info !19, !taffo.abserror !21
; CHECK: ret i32 %retval.0, !taffo.abserror !21

define i32 @bar(i32 %a, i32 %b) !taffo.funinfo !15 {
entry:
  %cmp = icmp sgt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %sub = sub nsw i32 %a, %b, !taffo.info !9
  br label %return

if.else:                                          ; preds = %entry
  %sub1 = sub nsw i32 %b, %a, !taffo.info !9
  br label %return

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ %sub, %if.then ], [ %sub1, %if.else ], !taffo.info !9
  ret i32 %retval.0
}

define i32 @foo(i32 %a, i32 %b) !taffo.funinfo !0 {
entry:
  %add = add nsw i32 %a, %b, !taffo.info !7
  %call = call i32 @bar(i32 %add, i32 %b), !taffo.info !9
  %cmp = icmp sgt i32 %call, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %add1 = add nsw i32 %add, %call, !taffo.info !11
  br label %return

if.end:                                           ; preds = %entry
  %sub = sub nsw i32 0, %call, !taffo.info !9
  %mul = mul nsw i32 %add, %sub, !taffo.info !13
  br label %return

return:                                           ; preds = %if.end, %if.then
  %retval.0 = phi i32 [ %add1, %if.then ], [ %mul, %if.end ], !taffo.info !13
  ret i32 %retval.0
}

!0 = !{i32 1, !1, i32 1, !5}
!1 = !{!2, !3, !4}
!2 = !{!"fixp", i32 -32, i32 6}
!3 = !{double -5.000000e+00, double 5.000000e+00}
!4 = !{double 1.000000e-02}
!5 = !{!2, !6, !4}
!6 = !{double -6.000000e+00, double 6.000000e+00}
!7 = !{!2, !8, i1 0}
!8 = !{double -1.100000e+01, double 1.100000e+01}
!9 = !{!2, !10, i1 0}
!10 = !{double -1.700000e+01, double 1.700000e+01}
!11 = !{!2, !12, i1 0}
!12 = !{double -2.800000e+01, double 2.800000e+01}
!13 = !{!2, !14, i1 0}
!14 = !{double -1.870000e+02, double 1.870000e+02}
!15 = !{i32 1, !16, i32 1, !17}
!16 = !{!2, !8, !4}
!17 = !{!2, !6, !4}

; CHECK: !7 = !{double 1.562500e-02}
; CHECK: !10 = !{double 2.000000e-02}
; CHECK: !17 = !{double 3.000000e-02}
; CHECK: !18 = !{double 0x3F9A3D70A3D70A3E}
; CHECK: !21 = !{double 0x3FE3EA9930BE0DED}
