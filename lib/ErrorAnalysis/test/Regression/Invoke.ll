; RUN: opt -load %errorproplib --load-pass-plugin=%errorproplib --passes="taffoerr" -S %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


; Function Attrs: noinline nounwind optnone uwtable
define i64 @_Z3barl(i64 %x) #0 !taffo.funinfo !0 {
entry:
  %x.addr = alloca i64, align 8
  store i64 %x, i64* %x.addr, align 8
  %0 = load i64, i64* %x.addr, align 8
  %1 = load i64, i64* %x.addr, align 8
  %mul = mul nsw i64 %0, %1, !taffo.info !5
  %2 = load i64, i64* %x.addr, align 8
  %mul1 = mul nsw i64 %mul, %2, !taffo.info !7
  ret i64 %mul1
}

; Function Attrs: noinline optnone uwtable
define i64 @_Z3fooll(i64 %a, i64 %b) #1 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) !taffo.funinfo !9 {
entry:
  %retval = alloca i64, align 8
  %a.addr = alloca i64, align 8
  %b.addr = alloca i64, align 8
  %exn.slot = alloca i8*
  %ehselector.slot = alloca i32
  store i64 %a, i64* %a.addr, align 8
  store i64 %b, i64* %b.addr, align 8
  %0 = load i64, i64* %a.addr, align 8
  %1 = load i64, i64* %b.addr, align 8
  %mul = mul nsw i64 %0, %1, !taffo.info !16
  %call = invoke i64 @_Z3barl(i64 %mul)
          to label %invoke.cont unwind label %lpad, !taffo.info !7

invoke.cont:                                      ; preds = %entry
  store i64 %call, i64* %retval, align 8
  br label %return

lpad:                                             ; preds = %entry
  %2 = landingpad { i8*, i32 }
          catch i8* null
  %3 = extractvalue { i8*, i32 } %2, 0
  store i8* %3, i8** %exn.slot, align 8
  %4 = extractvalue { i8*, i32 } %2, 1
  store i32 %4, i32* %ehselector.slot, align 4
  br label %catch

catch:                                            ; preds = %lpad
  %exn = load i8*, i8** %exn.slot, align 8
  %5 = call i8* @__cxa_begin_catch(i8* %exn) #3
  %6 = load i64, i64* %a.addr, align 8
  %7 = load i64, i64* %b.addr, align 8
  %mul1 = mul nsw i64 %6, %7, !taffo.info !16
  store i64 %mul1, i64* %retval, align 8
  call void @__cxa_end_catch()
  br label %return

try.cont:                                         ; No predecessors!
  call void @llvm.trap()
  unreachable

return:                                           ; preds = %catch, %invoke.cont
  %8 = load i64, i64* %retval, align 8
  ret i64 %8
}

declare i32 @__gxx_personality_v0(...)

declare i8* @__cxa_begin_catch(i8*)

declare void @__cxa_end_catch()

; Function Attrs: noreturn nounwind
declare void @llvm.trap() #2

attributes #0 = { noinline nounwind optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline optnone uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { noreturn nounwind }
attributes #3 = { nounwind }

!0 = !{i32 1, !1}
!1 = !{!2, !3, !4}
!2 = !{!"fixp", i32 64, i32 32}
!3 = !{double 3.000000e+00, double 1.100000e+01}
!4 = !{double 4.000000e-10}
!5 = !{!2, !6, i1 false}
!6 = !{double 9.000000e+00, double 1.210000e+02}
!7 = !{!2, !8, i1 false}
!8 = !{double 2.700000e+01, double 1.331000e+03}
!9 = !{i32 1, !10, i32 1, !13}
!10 = !{!2, !11, !12}
!11 = !{double 0.000000e+00, double 2.000000e+01}
!12 = !{double 5.000000e-10}
!13 = !{!2, !14, !15}
!14 = !{double 1.000000e+00, double 2.000000e+00}
!15 = !{double 6.000000e-10}
!16 = !{!2, !17, i1 false}
!17 = !{double 0.000000e+00, double 4.000000e+01}

; CHECK-DAG: !{double 0x3E42E5D9E5C5CC3B}
; CHECK-DAG: !{double 0x3E837D08B4F58035}
; CHECK-DAG: !{double 0x3E4BEAD35941E10C}
; CHECK-DAG: !{double 0x3ED3CAFCD82CAC6E}

;  store i64 %x, i64* %x.addr, align 8, !taffo.abserror !4
;  %0 = load i64, i64* %x.addr, align 8, !taffo.abserror !4
;  %1 = load i64, i64* %x.addr, align 8, !taffo.abserror !4
;  %mul = mul nsw i64 %0, %1, !taffo.info !5, !taffo.abserror !7
;  %2 = load i64, i64* %x.addr, align 8, !taffo.abserror !4
;  %mul1 = mul nsw i64 %mul, %2, !taffo.info !8, !taffo.abserror !10
;  ret i64 %mul1, !taffo.abserror !10
;  store i64 %a, i64* %a.addr, align 8, !taffo.abserror !14
;  store i64 %b, i64* %b.addr, align 8, !taffo.abserror !17
;  %0 = load i64, i64* %a.addr, align 8, !taffo.abserror !14
;  %1 = load i64, i64* %b.addr, align 8, !taffo.abserror !17
;  %mul = mul nsw i64 %0, %1, !taffo.info !18, !taffo.abserror !20
;  to label %invoke.cont unwind label %lpad, !taffo.info !8, !taffo.abserror !21
;  store i64 %call, i64* %retval, align 8, !taffo.abserror !21
;  %6 = load i64, i64* %a.addr, align 8, !taffo.abserror !14
;  %7 = load i64, i64* %b.addr, align 8, !taffo.abserror !17
;  %mul1 = mul nsw i64 %6, %7, !taffo.info !18, !taffo.abserror !20
;  store i64 %mul1, i64* %retval, align 8, !taffo.abserror !20
;  !7 = !{double 0x3E42E5D9E5C5CC3B}
;  !10 = !{double 0x3E837D08B4F58035}
;  !20 = !{double 0x3E4BEAD35941E10C}
;  !21 = !{double 0x3ED3CAFCD82CAC6E}
