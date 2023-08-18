; RUN: opt -load %errorproplib --load-pass-plugin=%errorproplib --passes="taffoerr" -dunroll 5 -S %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


; Function Attrs: noinline uwtable
define i32 @foo(i32 %a, i32 %b) #0 !taffo.funinfo !2 {
entry:
  %cmp1 = icmp slt i32 0, %b
  br i1 %cmp1, label %for.body.lr.ph, label %for.end

for.body.lr.ph:                                   ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.lr.ph, %for.body
  %a.addr.03 = phi i32 [ %a, %for.body.lr.ph ], [ %add, %for.body ], !taffo.info !7
  %i.02 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.body ]
  %add = add nsw i32 %a.addr.03, %a.addr.03, !taffo.info !7
  %inc = add nuw nsw i32 %i.02, 1
  %exitcond = icmp ne i32 %inc, %b
  br i1 %exitcond, label %for.body, label %for.cond.for.end_crit_edge

for.cond.for.end_crit_edge:                       ; preds = %for.body
  %split = phi i32 [ %add, %for.body ], !taffo.info !7
  br label %for.end

for.end:                                          ; preds = %for.cond.for.end_crit_edge, %entry
  %a.addr.0.lcssa = phi i32 [ %split, %for.cond.for.end_crit_edge ], [ %a, %entry ], !taffo.info !7
  %mul = mul nsw i32 %a.addr.0.lcssa, %a.addr.0.lcssa, !taffo.info !9
  ret i32 %mul
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.1 (https://git.llvm.org/git/clang.git/ 0e746072ed897a85b4f533ab050b9f506941a097) (git@github.com:llvm-mirror/llvm.git f63e4a17fa8545617336f7ebf5a418fdeee4530b)"}
!2 = !{i32 1, !3, i32 0, i32 0}
!3 = !{!4, !5, !6}
!4 = !{!"fixp", i32 -32, i32 4}
!5 = !{double 5.000000e+00, double 6.000000e+00}
!6 = !{double 1.250000e-02}
!7 = !{!4, !8, i1 0}
!8 = !{double 2.500000e+01, double 3.600000e+01}
!9 = !{!4, !10, i1 0}
!10 = !{double 1.000000e+04, double 2.073600e+04}

; CHECK-DAG: !{double 2.500000e-02}
; CHECK-DAG: !{double 4.000000e-01}
; CHECK-DAG: !{double 2.896000e+01}

;  %a.addr.03 = phi i32 [ %a, %for.body.lr.ph ], [ %add, %for.body ], !taffo.info !7, !taffo.abserror !6
;  %add = add nsw i32 %a.addr.03, %a.addr.03, !taffo.info !7, !taffo.abserror !9
;  %split = phi i32 [ %add, %for.body ], !taffo.info !7, !taffo.abserror !10
;  %a.addr.0.lcssa = phi i32 [ %split, %for.cond.for.end_crit_edge ], [ %a, %entry ], !taffo.info !7, !taffo.abserror !10
;  %mul = mul nsw i32 %a.addr.0.lcssa, %a.addr.0.lcssa, !taffo.info !11, !taffo.abserror !13
;  ret i32 %mul, !taffo.abserror !13
;  !9 = !{double 2.500000e-02}
;  !10 = !{double 4.000000e-01}
;  !13 = !{double 2.896000e+01}

