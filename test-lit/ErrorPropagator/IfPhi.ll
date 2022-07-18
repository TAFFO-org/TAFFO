; RUN: opt -load %errorproplib --load-pass-plugin=%errorproplib --passes="taffoerr" -S %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


; Function Attrs: noinline nounwind uwtable
define i32 @foo(i32 %a, i32 %b) #0 !taffo.funinfo !3 {
entry:
  %cmp = icmp slt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %add = add nsw i32 %a, %b, !taffo.info !11
  br label %if.end

if.else:                                          ; preds = %entry
  %sub = sub nsw i32 %a, %b, !taffo.info !13
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %a.addr.0 = phi i32 [ %add, %if.then ], [ %sub, %if.else ], !taffo.info !15
  %mul = mul nsw i32 %a.addr.0, %b, !taffo.info !17
  %div = sdiv i32 %b, %mul, !taffo.info !19
  ret i32 %div
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.1 (https://git.llvm.org/git/clang.git/ 0e746072ed897a85b4f533ab050b9f506941a097) (git@github.com:llvm-mirror/llvm.git ce53c20d527634abbccce7caf92891517ba0ab30)"}
!3 = !{i32 1, !4, i32 1, !8}
!4 = !{!5, !6, !7}
!5 = !{!"fixp", i32 -32, i32 3}
!6 = !{double 1.000000e+00, double 1.500000e+00}
!7 = !{double 1.250000e-01}
!8 = !{!5, !9, !10}
!9 = !{double 1.250000e+00, double 1.750000e+00}
!10 = !{double 2.000000e-03}
!11 = !{!5, !12, i1 0}
!12 = !{double 2.250000e+00, double 3.250000e+00}
!13 = !{!5, !14, i1 0}
!14 = !{double -7.500000e-01, double 2.500000e-01}
!15 = !{!5, !16, i1 0}
!16 = !{double -7.500000e-01, double 3.250000e+00}
!17 = !{!5, !18, i1 0}
!18 = !{double -1.500000e+01, double 4.550000e+01}
!19 = !{!5, !20, i1 0}
!20 = !{double -1.000000e+00, double 3.250000e+00}

; CHECK-DAG: !{double 1.270000e-01}
; CHECK-DAG: !{double 2.290040e-01}
; CHECK-DAG: !{double 0x3FC03ECCDC5942DB}

;  %cmp = icmp slt i32 %a, %b, !taffo.wrongcmptol !6
;  %add = add nsw i32 %a, %b, !taffo.info !10, !taffo.abserror !12
;  %sub = sub nsw i32 %a, %b, !taffo.info !13, !taffo.abserror !12
;  %a.addr.0 = phi i32 [ %add, %if.then ], [ %sub, %if.else ], !taffo.info !15, !taffo.abserror !12
;  %mul = mul nsw i32 %a.addr.0, %b, !taffo.info !17, !taffo.abserror !19
;  %div = sdiv i32 %b, %mul, !taffo.info !20, !taffo.abserror !22
;  ret i32 %div, !taffo.abserror !22
;  !12 = !{double 1.270000e-01}
;  !19 = !{double 2.290040e-01}
;  !22 = !{double 0x3FC03ECCDC5942DB}
