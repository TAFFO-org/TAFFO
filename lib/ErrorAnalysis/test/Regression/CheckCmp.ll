; RUN: opt -load %errorproplib -errorprop -cmpthresh 25 -S %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: %cmp = icmp slt i32 %a, %b, !taffo.wrongcmptol !6
; CHECK: %sub = sub nsw i32 %b, %a, !taffo.info !9, !taffo.abserror !11
; CHECK: %sub1 = sub nsw i32 %a, %b, !taffo.info !12, !taffo.abserror !11
; CHECK: %c.0 = phi i32 [ %sub, %if.then ], [ %sub1, %if.else ], !taffo.info !14, !taffo.abserror !11
; CHECK: ret i32 %c.0, !taffo.abserror !11

; Function Attrs: noinline uwtable
define i32 @bar(i32 %a, i32 %b) !taffo.funinfo !2 {
entry:
  %cmp = icmp slt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %sub = sub nsw i32 %b, %a, !taffo.info !9
  br label %if.end

if.else:                                          ; preds = %entry
  %sub1 = sub nsw i32 %a, %b, !taffo.info !11
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %c.0 = phi i32 [ %sub, %if.then ], [ %sub1, %if.else ], !taffo.info !13
  ret i32 %c.0
}

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.1 (https://git.llvm.org/git/clang.git/ 0e746072ed897a85b4f533ab050b9f506941a097) (git@github.com:llvm-mirror/llvm.git 38c1684af2387229b58d2ca8c57202ed3e60e1e3)"}
!2 = !{i32 1, !3, i32 1, !4}
!3 = !{!5, !6, !7}
!5 = !{!"fixp", i32 -32, i32 5}
!6 = !{double 4.000000e+00, double 5.000000e+00}
!7 = !{double 1.000000e+00}
!4 = !{!5, !8, !7}
!8 = !{double 6.000000e+00, double 7.000000e+00}
!9 = !{!5, !10, i1 0}
!10 = !{double 2.000000e+00, double 2.000000e+00}
!11 = !{!5, !12, i1 0}
!12 = !{double -2.000000e+00, double -2.000000e+00}
!13 = !{!5, !14, i1 0}
!14 = !{double -2.000000e+00, double 2.000000e+00}

; CHECK: !11 = !{double 2.000000e+00}
