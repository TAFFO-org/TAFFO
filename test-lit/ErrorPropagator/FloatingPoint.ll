; RUN: opt -load %errorproplib --load-pass-plugin=%errorproplib --passes="taffoerr" -S %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: %add = add nsw i64 %a, %b, !taffo.info !7, !taffo.abserror !9
; CHECK: %conv = sitofp i64 %add to float, !taffo.abserror !9
; CHECK: %conv1 = sitofp i64 %b to float, !taffo.abserror !4
; CHECK: %mul = fmul float %conv, %conv1, !taffo.info !10, !taffo.abserror !12
; CHECK: %conv2 = fpext float %mul to double, !taffo.abserror !12
; CHECK: %conv3 = sitofp i64 %a to double, !taffo.abserror !4
; CHECK: %sub = fsub double %conv2, %conv3, !taffo.info !13, !taffo.abserror !15
; CHECK: %conv4 = fptrunc double %sub to float, !taffo.abserror !15
; CHECK: %conv6 = fptosi float %conv4 to i32, !taffo.info !16, !taffo.abserror !18
; CHECK: %conv7 = fptosi float %conv to i32, !taffo.info !19, !taffo.abserror !20
; CHECK: %retval.0 = phi i32 [ %conv6, %if.then ], [ %conv7, %if.else ], !taffo.info !16, !taffo.abserror !18
; CHECK: ret i32 %retval.0, !taffo.abserror !18

define i32 @foo(i64 %a, i64 %b) #0 !taffo.funinfo !0 {
entry:
  %add = add nsw i64 %a, %b, !taffo.info !7
  %conv = sitofp i64 %add to float
  %conv1 = sitofp i64 %b to float
  %mul = fmul float %conv, %conv1, !taffo.info !9
  %conv2 = fpext float %mul to double
  %conv3 = sitofp i64 %a to double
  %sub = fsub double %conv2, %conv3, !taffo.info !11
  %conv4 = fptrunc double %sub to float
  %cmp = fcmp ogt float %conv, %conv4
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %conv6 = fptosi float %conv4 to i32, !taffo.info !13
  br label %return

if.else:                                          ; preds = %entry
  %conv7 = fptosi float %conv to i32, !taffo.info !15
  br label %return

return:                                           ; preds = %if.else, %if.then
  %retval.0 = phi i32 [ %conv6, %if.then ], [ %conv7, %if.else ], !taffo.info !13
  ret i32 %retval.0
}

!0 = !{i32 1, !1, i32 1, !5}
!1 = !{!2, !3, !4}
!2 = !{!"fixp", i32 -64, i32 20}
!3 = !{double 1.000000e+01, double 5.000000e+01}
!4 = !{double 5.000000e-06}
!5 = !{!2, !6, !4}
!6 = !{double 1.000000e+02, double 2.000000e+02}
!7 = !{!2, !8, i1 0}
!8 = !{double 1.100000e+02, double 2.500000e+02}
!9 = !{i1 0, !10, i1 0}
!10 = !{double 1.100000e+04, double 5.000000e+04}
!11 = !{i1 0, !12, i1 0}
!12 = !{double 1.099000e+04, double 4.995000e+04}
!13 = !{!14, !12, i1 0}
!14 = !{!"fixp", i32 -32, i32 20}
!15 = !{!14, !8, i1 0}

; CHECK: !4 = !{double 5.000000e-06}
; CHECK: !9 = !{double 1.000000e-05}
; CHECK: !12 = !{double 0x3F6A9FBE7DA7EC30}
; CHECK: !15 = !{double 0x3F6AAA3AD86C5DE5}
; CHECK: !18 = !{double 0x3F6AAC3AD86C5DE5}
