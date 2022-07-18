; RUN: opt -load %errorproplib --load-pass-plugin=%errorproplib --passes="taffoerr" -S %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: %a.addr.02 = phi i32 [ %a, %entry ], [ %add, %for.body ], !taffo.abserror !6
; CHECK: %add = add nsw i32 %a.addr.02, %a.addr.02, !taffo.info !7, !taffo.abserror !9
; CHECK: %a.addr.0.lcssa = phi i32 [ %add, %for.body ], !taffo.abserror !10
; CHECK: %mul = mul nsw i32 %a.addr.0.lcssa, %a.addr.0.lcssa, !taffo.info !11, !taffo.abserror !13
; CHECK: ret i32 %mul, !taffo.abserror !13

; Function Attrs: noinline uwtable
define i32 @foo(i32 %a) #0 !taffo.funinfo !2 {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %a.addr.02 = phi i32 [ %a, %entry ], [ %add, %for.body ]
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %add = add nsw i32 %a.addr.02, %a.addr.02, !taffo.info !7
  %inc = add nuw nsw i32 %i.01, 1
  %exitcond = icmp ne i32 %inc, 10
  br i1 %exitcond, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %a.addr.0.lcssa = phi i32 [ %add, %for.body ]
  %mul = mul nsw i32 %a.addr.0.lcssa, %a.addr.0.lcssa, !taffo.info !9
  ret i32 %mul
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.1 (https://git.llvm.org/git/clang.git/ 0e746072ed897a85b4f533ab050b9f506941a097) (git@github.com:llvm-mirror/llvm.git 7883f391cb5539d062f0d6d9b3aa05b159b18450)"}
!2 = !{i32 1, !3}
!3 = !{!4, !5, !6}
!4 = !{!"fixp", i32 -32, i32 4}
!5 = !{double 5.000000e+00, double 6.000000e+00}
!6 = !{double 1.250000e-02}
!7 = !{!4, !8, i1 0}
!8 = !{double 5.000000e+01, double 6.000000e+01}
!9 = !{!4, !10, i1 0}
!10 = !{double 2.500000e+03, double 3.600000e+03}

; CHECK: !9 = !{double 2.500000e-02}
; CHECK: !10 = !{double 1.280000e+01}
; CHECK: !13 = !{double 0x409A8F5C28F5C290}
