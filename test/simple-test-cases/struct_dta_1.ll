;;;TAFFO_TEST_ARGS -disable-vra
; ModuleID = 'struct_integration_1.out.1.magiclangtmp.ll'
source_filename = "struct_integration_1.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

%struct.test = type { float, i32, float }

@.str = private unnamed_addr constant [65 x i8] c"struct(scalar(range(-3000, +3000)) void() scalar(range(-3, +3)))\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [23 x i8] c"struct_integration_1.c\00", section "llvm.metadata"
@.str.2 = private unnamed_addr constant [7 x i8] c"%f%f%f\00", align 1
@.str.3 = private unnamed_addr constant [10 x i8] c"%f\0A%d\0A%f\0A\00", align 1

; Function Attrs: noinline nounwind ssp uwtable
define i32 @main(i32 %argc, i8** %argv) #0 !taffo.funinfo !3 {
entry:
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %z = alloca %struct.test, align 4, !taffo.structinfo !4
  %a = alloca float, align 4
  %b = alloca float, align 4
  %c = alloca float, align 4
  
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  store i8** %argv, i8*** %argv.addr, align 8

  %call = call i32 (i8*, ...) @scanf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.2, i32 0, i32 0), float* %a, float* %b, float* %c)
  
  %0 = load float, float* %a, align 4
  %a2 = getelementptr inbounds %struct.test, %struct.test* %z, i32 0, i32 0, !taffo.info !5
  store float %0, float* %a2, align 4, !taffo.info !5
  
  %1 = load float, float* %b, align 4
  %conv = fptosi float %1 to i32
  %b3 = getelementptr inbounds %struct.test, %struct.test* %z, i32 0, i32 1, !taffo.info !9
  store i32 %conv, i32* %b3, align 4, !taffo.info !9
  
  %2 = load float, float* %c, align 4
  %c4 = getelementptr inbounds %struct.test, %struct.test* %z, i32 0, i32 2, !taffo.info !7
  store float %2, float* %c4, align 4, !taffo.info !7
  
  %a5 = getelementptr inbounds %struct.test, %struct.test* %z, i32 0, i32 0, !taffo.info !5
  %3 = load float, float* %a5, align 4, !taffo.info !5
  %conv6 = fpext float %3 to double, !taffo.info !5
  
  %b7 = getelementptr inbounds %struct.test, %struct.test* %z, i32 0, i32 1, !taffo.info !9
  %4 = load i32, i32* %b7, align 4, !taffo.info !9
  
  %c8 = getelementptr inbounds %struct.test, %struct.test* %z, i32 0, i32 2, !taffo.info !7
  %5 = load float, float* %c8, align 4, !taffo.info !7
  %conv9 = fpext float %5 to double, !taffo.info !7
  
  %call10 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.3, i32 0, i32 0), double %conv6, i32 %4, double %conv9), !taffo.info !9
  ret i32 0
}

; Function Attrs: nounwind
declare !taffo.funinfo !10 void @llvm.var.annotation(i8*, i8*, i8*, i32) #1

declare !taffo.funinfo !11 i32 @scanf(i8*, ...) #2

declare !taffo.funinfo !11 i32 @printf(i8*, ...) #2

attributes #0 = { noinline nounwind ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sahf,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{!"clang version 6.0.1 (tags/RELEASE_601/final)"}
!3 = !{i32 0, i1 false, i32 0, i1 false}
!4 = !{!5, i1 false, !7}
!5 = !{i1 false, !6, i1 false, i1 true}
!6 = !{double -3.000000e+03, double 3.000000e+03}
!7 = !{i1 false, !8, i1 false, i1 true}
!8 = !{double -3.000000e+00, double 3.000000e+00}
!9 = !{i1 false, i1 false, i1 false, i1 true}
!10 = !{i32 0, i1 false, i32 0, i1 false, i32 0, i1 false, i32 0, i1 false}
!11 = !{i32 0, i1 false}
