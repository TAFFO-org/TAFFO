; RUN: opt -load %errorproplib --load-pass-plugin=%errorproplib --passes="taffoerr" -S %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: %0 = sub i32 0, %b, !taffo.info !10, !taffo.abserror !12
; CHECK: %a.addr.0.p = select i1 %cmp, i32 %b, i32 %0, !taffo.info !13, !taffo.abserror !12
; CHECK: %a.addr.0 = add i32 %a.addr.0.p, %a, !taffo.info !15, !taffo.abserror !17
; CHECK: %mul = mul nsw i32 %a.addr.0, %b, !taffo.info !18, !taffo.abserror !20
; CHECK: %div = sdiv i32 %mul, %b, !taffo.info !21, !taffo.abserror !23
; CHECK: ret i32 %div, !taffo.abserror !23

; Function Attrs: norecurse nounwind readnone uwtable
define i32 @foo(i32 %a, i32 %b) local_unnamed_addr #0 !taffo.funinfo !3 {
entry:
  %cmp = icmp slt i32 %a, %b
  %0 = sub i32 0, %b, !taffo.info !11
  %a.addr.0.p = select i1 %cmp, i32 %b, i32 %0, !taffo.info !13
  %a.addr.0 = add i32 %a.addr.0.p, %a, !taffo.info !15
  %mul = mul nsw i32 %a.addr.0, %b, !taffo.info !17
  %div = sdiv i32 %mul, %b, !taffo.info !19
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
!7 = !{double 1.000000e-02}
!8 = !{!5, !9, !10}
!9 = !{double 1.250000e+00, double 1.750000e+00}
!10 = !{double 2.000000e-03}
!11 = !{!5, !12, i1 false}
!12 = !{double -1.750000e+00, double -1.250000e+00}
!13 = !{!5, !14, i1 false}
!14 = !{double -1.250000e+00, double 1.750000e+00}
!15 = !{!5, !16, i1 false}
!16 = !{double -0.750000e+00, double 3.250000e+00}
!17 = !{!5, !18, i1 false}
!18 = !{double 1.050000e+01, double 4.550000e+01}
!19 = !{!5, !20, i1 false}
!20 = !{double -1.000000e+00, double 3.250000e+00}

; CHECK: !12 = !{double 1.270000e-01}
; CHECK: !17 = !{double 1.370000e-01}
; CHECK: !20 = !{double 2.465240e-01}
; CHECK: !23 = !{double 0x3FD85E9D0F5F0976}
