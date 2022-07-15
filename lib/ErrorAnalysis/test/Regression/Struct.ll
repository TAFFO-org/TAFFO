; RUN: opt -load %errorproplib --load-pass-plugin=%errorproplib --passes="require<globals-aa>,function(require<cfl-steens-aa>,require<cfl-anders-aa>,require<tbaa>),taffoerr" -S %s | FileCheck %s


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.S = type { i32, i64 }

define i32 @foo(i32 %s.coerce0, i64 %s.coerce1, i64 %c) !taffo.funinfo !0 {
entry:
  %s = alloca %struct.S, align 8
  %c.addr = alloca i64, align 8
  %0 = bitcast %struct.S* %s to { i32, i64 }*
  %1 = getelementptr inbounds { i32, i64 }, { i32, i64 }* %0, i32 0, i32 0
  store i32 %s.coerce0, i32* %1, align 8
  %2 = getelementptr inbounds { i32, i64 }, { i32, i64 }* %0, i32 0, i32 1
  store i64 %s.coerce1, i64* %2, align 8
  store i64 %c, i64* %c.addr, align 8
  %3 = load i64, i64* %c.addr, align 8
  %a = getelementptr inbounds %struct.S, %struct.S* %s, i32 0, i32 0
  %4 = load i32, i32* %a, align 8
  %conv = sext i32 %4 to i64
  %add = add nsw i64 %conv, %3, !taffo.info !9
  %conv1 = trunc i64 %add to i32
  store i32 %conv1, i32* %a, align 8
  %5 = load i64, i64* %c.addr, align 8
  %b = getelementptr inbounds %struct.S, %struct.S* %s, i32 0, i32 1
  %6 = load i64, i64* %b, align 8, !taffo.info !5
  %mul = mul nsw i64 %6, %5, !taffo.info !11
  store i64 %mul, i64* %b, align 8
  %a2 = getelementptr inbounds %struct.S, %struct.S* %s, i32 0, i32 0
  %7 = load i32, i32* %a2, align 8, !taffo.info !9
  %conv3 = sext i32 %7 to i64
  %b4 = getelementptr inbounds %struct.S, %struct.S* %s, i32 0, i32 1
  %8 = load i64, i64* %b4, align 8, !taffo.info !11
  %add5 = add nsw i64 %conv3, %8, !taffo.info !13
  %conv6 = trunc i64 %add5 to i32
  ret i32 %conv6
}

!0 = !{i32 1, !1, i32 1, !5, i32 1, !7}
!1 = !{!15, !3, !4}
!2 = !{!"fixp", i32 -64, i32 15}
!3 = !{double -5.000000e+00, double 1.000000e+01}
!4 = !{double 2.000000e-05}
!5 = !{!2, !6, !4}
!6 = !{double -1.000000e+01, double 2.000000e+01}
!7 = !{!2, !8, !4}
!8 = !{double -2.000000e+01, double 1.500000e+01}
!9 = !{!2, !10, i1 0}
!10 = !{double -2.500000e+01, double 2.500000e+01}
!11 = !{!2, !12, i1 0}
!12 = !{double -4.000000e+02, double 3.000000e+02}
!13 = !{!2, !14, i1 0}
!14 = !{double -4.250000e+02, double 3.250000e+02}
!15 = !{!"fixp", i32 -32, i32 15}

; CHECK-DAG: !{double 0x3F4A36E3C70341FC}
; CHECK-DAG: !{double 4.000000e-05}
; CHECK-DAG: !{double 0x3F4B866F1F91788B}

;  store i32 %s.coerce0, i32* %1, align 8, !taffo.abserror !4
;  store i64 %s.coerce1, i64* %2, align 8, !taffo.abserror !4
;  store i64 %c, i64* %c.addr, align 8, !taffo.abserror !4
;  %3 = load i64, i64* %c.addr, align 8, !taffo.abserror !4
;  %4 = load i32, i32* %a, align 8, !taffo.abserror !4
;  %conv = sext i32 %4 to i64, !taffo.abserror !4
;  %add = add nsw i64 %conv, %3, !taffo.info !10, !taffo.abserror !12
;  %conv1 = trunc i64 %add to i32, !taffo.abserror !12
;  store i32 %conv1, i32* %a, align 8, !taffo.abserror !12
;  %5 = load i64, i64* %c.addr, align 8, !taffo.abserror !4
;  %6 = load i64, i64* %b, align 8, !taffo.abserror !4
;  %mul = mul nsw i64 %6, %5, !taffo.info !13, !taffo.abserror !15
;  store i64 %mul, i64* %b, align 8, !taffo.abserror !15
;  %7 = load i32, i32* %a2, align 8, !taffo.abserror !12
;  %conv3 = sext i32 %7 to i64, !taffo.abserror !12
;  %8 = load i64, i64* %b4, align 8, !taffo.abserror !15
;  %add5 = add nsw i64 %conv3, %8, !taffo.info !16, !taffo.abserror !18
;  %conv6 = trunc i64 %add5 to i32, !taffo.abserror !18
;  ret i32 %conv6, !taffo.abserror !18
;  !12 = !{double 4.000000e-05}
;  !15 = !{double 0x3F4A36E3C70341FC}
;  !18 = !{double 0x3F4B866F1F91788B}
