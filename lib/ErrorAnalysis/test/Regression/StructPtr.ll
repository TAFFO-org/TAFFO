; RUN:  opt -load %errorproplib -errorprop -S %s | FileCheck %s

%struct.bar = type { i32, i64 }
%struct.foo = type { %struct.bar, %struct.bar*, i64, [5 x [4 x i32]] }

@globby = common global %struct.bar zeroinitializer, align 8, !taffo.structinfo !10

; Function Attrs: noinline nounwind uwtable
define i32 @slarti(%struct.foo* noalias nocapture, i32) #0 !taffo.funinfo !7 {
  %3 = sext i32 %1 to i64, !taffo.info !2
  %4 = getelementptr inbounds %struct.foo, %struct.foo* %0, i32 0, i32 0
  %5 = getelementptr inbounds %struct.bar, %struct.bar* %4, i32 0, i32 1
  store i64 %3, i64* %5, align 8
  %6 = getelementptr inbounds %struct.foo, %struct.foo* %0, i32 0, i32 0
  %7 = getelementptr inbounds %struct.bar, %struct.bar* %6, i32 0, i32 1
  %8 = load i64, i64* %7, align 8, !taffo.info !6
  %9 = sext i32 %1 to i64, !taffo.info !6
  %10 = add nsw i64 %8, %9, !taffo.info !6
  %11 = getelementptr inbounds %struct.foo, %struct.foo* %0, i32 0, i32 1
  %12 = load %struct.bar*, %struct.bar** %11, align 8
  %13 = getelementptr inbounds %struct.bar, %struct.bar* %12, i32 0, i32 1
  store i64 %10, i64* %13, align 8
  %14 = getelementptr inbounds %struct.foo, %struct.foo* %0, i32 0, i32 0
  %15 = getelementptr inbounds %struct.bar, %struct.bar* %14, i32 0, i32 1
  %16 = load i64, i64* %15, align 8, !taffo.info !6
  %17 = getelementptr inbounds %struct.foo, %struct.foo* %0, i32 0, i32 1
  %18 = load %struct.bar*, %struct.bar** %17, align 8
  %19 = getelementptr inbounds %struct.bar, %struct.bar* %18, i32 0, i32 1
  %20 = load i64, i64* %19, align 8, !taffo.info !6
  %21 = add nsw i64 %16, %20, !taffo.info !6
  %22 = trunc i64 %21 to i32, !taffo.info !6
  ret i32 %22, !taffo.info !6
}

; Function Attrs: noinline nounwind uwtable
define i32 @main() #0 {
  %1 = alloca %struct.foo, align 8, !taffo.structinfo !13
  %2 = alloca %struct.bar, align 8, !taffo.structinfo !10
  %3 = trunc i64 42 to i32, !taffo.info !8
  store i32 %3, i32* getelementptr inbounds (%struct.bar, %struct.bar* @globby, i32 0, i32 0), align 8
  %4 = getelementptr inbounds %struct.foo, %struct.foo* %1, i32 0, i32 1
  store %struct.bar* %2, %struct.bar** %4, align 8
  %5 = call i32 @slarti(%struct.foo* %1, i32 15)
  %6 = getelementptr inbounds %struct.foo, %struct.foo* %1, i32 0, i32 0
  %7 = getelementptr inbounds %struct.bar, %struct.bar* %6, i32 0, i32 1
  %8 = load i64, i64* %7, align 8, !taffo.info !6
  %9 = trunc i64 %8 to i32, !taffo.info !6
  %10 = getelementptr inbounds %struct.foo, %struct.foo* %1, i32 0, i32 3
  %11 = getelementptr inbounds [5 x [4 x i32]], [5 x [4 x i32]]* %10, i64 0, i64 2
  %12 = getelementptr inbounds [4 x i32], [4 x i32]* %11, i64 0, i64 3
  store i32 %9, i32* %12, align 4
  %13 = getelementptr inbounds %struct.foo, %struct.foo* %1, i32 0, i32 3
  %14 = getelementptr inbounds [5 x [4 x i32]], [5 x [4 x i32]]* %13, i64 0, i64 2
  %15 = getelementptr inbounds [4 x i32], [4 x i32]* %14, i64 0, i64 3
  %16 = load i32, i32* %15, align 4, !taffo.info !6
  %17 = getelementptr inbounds %struct.foo, %struct.foo* %1, i32 0, i32 3
  %18 = getelementptr inbounds [5 x [4 x i32]], [5 x [4 x i32]]* %17, i64 0, i64 1
  %19 = getelementptr inbounds [4 x i32], [4 x i32]* %18, i64 0, i64 0
  store i32 %16, i32* %19, align 8
  %20 = load i32, i32* getelementptr inbounds (%struct.bar, %struct.bar* @globby, i32 0, i32 0), align 8, !taffo.info !6
  %21 = sext i32 %20 to i64, !taffo.info !6
  %22 = getelementptr inbounds %struct.foo, %struct.foo* %1, i32 0, i32 0
  %23 = getelementptr inbounds %struct.bar, %struct.bar* %22, i32 0, i32 1
  %24 = load i64, i64* %23, align 8, !taffo.info !6
  %25 = add nsw i64 %21, %24, !taffo.info !6
  store i64 %25, i64* getelementptr inbounds (%struct.bar, %struct.bar* @globby, i32 0, i32 1), align 8
  %26 = getelementptr inbounds %struct.foo, %struct.foo* %1, i32 0, i32 1
  %27 = load %struct.bar*, %struct.bar** %26, align 8
  %28 = getelementptr inbounds %struct.bar, %struct.bar* %27, i32 0, i32 1
  %29 = load i64, i64* %28, align 8, !taffo.info !6
  %30 = trunc i64 %29 to i32, !taffo.info !6
  ret i32 %30, !taffo.info !6
}

attributes #0 = { noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 6.0.0-1ubuntu2 (tags/RELEASE_600/final)"}
!2 = !{!3, !4, !5}
!3 = !{!"fixp", i32 -32, i32 25}
!4 = !{double -2.000000e+00, double 2.000000e+00}
!5 = !{double 1.000000e-08}
!6 = !{!3, !4, i1 false}
!7 = !{i32 0, i32 0, i32 1, !2}
!8 = !{!3, !4, !9}
!9 = !{double 1.000000e-07}
!10 = !{!11, !6}
!11 = !{!3, !12, i1 false}
!12 = !{double 42.000000e+00, double 42.000000e+00}
!13 = !{!10, !10, i1 false, i1 false}

; CHECK-DAG: !{double 2.000000e-08}
; CHECK-DAG: !{double 0x3E601B2B29A4692C}
; CHECK-DAG: !{double 0x3E7AD7F29ABCAF48}
; CHECK-DAG: !{double 0x3E60000000000000}
; CHECK-DAG: !{double 0x3E6ABCC77118461D}

; ; Function Attrs: noinline nounwind uwtable
; define i32 @slarti(%struct.foo* noalias nocapture, i32) #0 !taffo.funinfo !2 {
;   %3 = sext i32 %1 to i64, !taffo.info !6, !taffo.abserror !7
;   %4 = getelementptr inbounds %struct.foo, %struct.foo* %0, i32 0, i32 0
;   %5 = getelementptr inbounds %struct.bar, %struct.bar* %4, i32 0, i32 1
;   store i64 %3, i64* %5, align 8, !taffo.abserror !7
;   %6 = getelementptr inbounds %struct.foo, %struct.foo* %0, i32 0, i32 0
;   %7 = getelementptr inbounds %struct.bar, %struct.bar* %6, i32 0, i32 1
;   %8 = load i64, i64* %7, align 8, !taffo.info !3, !taffo.abserror !7
;   %9 = sext i32 %1 to i64, !taffo.info !3, !taffo.abserror !7
;   %10 = add nsw i64 %8, %9, !taffo.info !3, !taffo.abserror !8
;   %11 = getelementptr inbounds %struct.foo, %struct.foo* %0, i32 0, i32 1
;   %12 = load %struct.bar*, %struct.bar** %11, align 8
;   %13 = getelementptr inbounds %struct.bar, %struct.bar* %12, i32 0, i32 1
;   store i64 %10, i64* %13, align 8, !taffo.abserror !8
;   %14 = getelementptr inbounds %struct.foo, %struct.foo* %0, i32 0, i32 0
;   %15 = getelementptr inbounds %struct.bar, %struct.bar* %14, i32 0, i32 1
;   %16 = load i64, i64* %15, align 8, !taffo.info !3, !taffo.abserror !7
;   %17 = getelementptr inbounds %struct.foo, %struct.foo* %0, i32 0, i32 1
;   %18 = load %struct.bar*, %struct.bar** %17, align 8
;   %19 = getelementptr inbounds %struct.bar, %struct.bar* %18, i32 0, i32 1
;   %20 = load i64, i64* %19, align 8, !taffo.info !3, !taffo.abserror !8
;   %21 = add nsw i64 %16, %20, !taffo.info !3, !taffo.abserror !9
;   %22 = trunc i64 %21 to i32, !taffo.info !3, !taffo.abserror !9
;   ret i32 %22, !taffo.info !3, !taffo.abserror !9
; }
; 
; ; Function Attrs: noinline nounwind uwtable
; define i32 @main() #0 {
;   %1 = alloca %struct.foo, align 8, !taffo.abserror !8
;   %2 = alloca %struct.bar, align 8
;   %3 = trunc i64 42 to i32, !taffo.info !10, !taffo.abserror !12
;   store i32 %3, i32* getelementptr inbounds (%struct.bar, %struct.bar* @globby, i32 0, i32 0), align 8, !taffo.abserror !12
;   %4 = getelementptr inbounds %struct.foo, %struct.foo* %1, i32 0, i32 1
;   store %struct.bar* %2, %struct.bar** %4, align 8
;   %5 = call i32 @slarti(%struct.foo* %1, i32 15)
;   %6 = getelementptr inbounds %struct.foo, %struct.foo* %1, i32 0, i32 0
;   %7 = getelementptr inbounds %struct.bar, %struct.bar* %6, i32 0, i32 1
;   %8 = load i64, i64* %7, align 8, !taffo.info !3, !taffo.abserror !8
;   %9 = trunc i64 %8 to i32, !taffo.info !3, !taffo.abserror !8
;   %10 = getelementptr inbounds %struct.foo, %struct.foo* %1, i32 0, i32 3
;   %11 = getelementptr inbounds [5 x [4 x i32]], [5 x [4 x i32]]* %10, i64 0, i64 2
;   %12 = getelementptr inbounds [4 x i32], [4 x i32]* %11, i64 0, i64 3
;   store i32 %9, i32* %12, align 4, !taffo.abserror !8
;   %13 = getelementptr inbounds %struct.foo, %struct.foo* %1, i32 0, i32 3
;   %14 = getelementptr inbounds [5 x [4 x i32]], [5 x [4 x i32]]* %13, i64 0, i64 2
;   %15 = getelementptr inbounds [4 x i32], [4 x i32]* %14, i64 0, i64 3
;   %16 = load i32, i32* %15, align 4, !taffo.info !3, !taffo.abserror !8
;   %17 = getelementptr inbounds %struct.foo, %struct.foo* %1, i32 0, i32 3
;   %18 = getelementptr inbounds [5 x [4 x i32]], [5 x [4 x i32]]* %17, i64 0, i64 1
;   %19 = getelementptr inbounds [4 x i32], [4 x i32]* %18, i64 0, i64 0
;   store i32 %16, i32* %19, align 8, !taffo.abserror !8
;   %20 = load i32, i32* getelementptr inbounds (%struct.bar, %struct.bar* @globby, i32 0, i32 0), align 8, !taffo.info !3, !taffo.abserror !12
;   %21 = sext i32 %20 to i64, !taffo.info !3, !taffo.abserror !12
;   %22 = getelementptr inbounds %struct.foo, %struct.foo* %1, i32 0, i32 0
;   %23 = getelementptr inbounds %struct.bar, %struct.bar* %22, i32 0, i32 1
;   %24 = load i64, i64* %23, align 8, !taffo.info !3, !taffo.abserror !8
;   %25 = add nsw i64 %21, %24, !taffo.info !3, !taffo.abserror !13
;   store i64 %25, i64* getelementptr inbounds (%struct.bar, %struct.bar* @globby, i32 0, i32 1), align 8, !taffo.abserror !13
;   %26 = getelementptr inbounds %struct.foo, %struct.foo* %1, i32 0, i32 1
;   %27 = load %struct.bar*, %struct.bar** %26, align 8
;   %28 = getelementptr inbounds %struct.bar, %struct.bar* %27, i32 0, i32 1
;   %29 = load i64, i64* %28, align 8, !taffo.info !3, !taffo.abserror !13
;   %30 = trunc i64 %29 to i32, !taffo.info !3, !taffo.abserror !13
;   ret i32 %30, !taffo.info !3, !taffo.abserror !13
; }
; !8 = !{double 2.000000e-08}
; !9 = !{double 0x3E601B2B29A4692C}
; !10 = !{!4, !5, !11}
; !11 = !{double 0x3E7AD7F29ABCAF48}
; !12 = !{double 0x3E60000000000000}
; !13 = !{double 0x3E6ABCC77118461D}
