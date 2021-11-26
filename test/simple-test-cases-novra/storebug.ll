;;;TAFFO_TEST_ARGS -disable-vra
; ModuleID = 'test.c'
source_filename = "test.c"
target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.12.0"

@.str = private unnamed_addr constant [19 x i8] c"range -32767 32767\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [7 x i8] c"test.c\00", section "llvm.metadata"
@main.test = private unnamed_addr constant [5 x float] [float 1.230000e+02, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00, float 0.000000e+00], align 4
@.str.2 = private unnamed_addr constant [3 x i8] c"%f\00", align 1

; Function Attrs: alwaysinline nounwind ssp
define void @hello(float* %abc) #0 {
entry:
  %abc.addr = alloca float*, align 4
  store float* %abc, float** %abc.addr, align 4
  %abc.addr1 = bitcast float** %abc.addr to i8*
  call void @llvm.var.annotation(i8* %abc.addr1, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.1, i32 0, i32 0), i32 4)
  %0 = load float*, float** %abc.addr, align 4
  %arrayidx = getelementptr inbounds float, float* %0, i32 1
  %1 = load float, float* %arrayidx, align 4
  %add = fadd float %1, 5.000000e+00
  store float %add, float* %arrayidx, align 4
  ret void
}

; Function Attrs: nounwind
declare void @llvm.var.annotation(i8*, i8*, i8*, i32) #1

; Function Attrs: noinline nounwind ssp
define i32 @main(i32 %argc, i8** %argv) #2 {
entry:
  %abc.addr.i = alloca float*, align 4
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 4
  %test = alloca [5 x float], align 4
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  store i8** %argv, i8*** %argv.addr, align 4
  %test1 = bitcast [5 x float]* %test to i8*
  call void @llvm.var.annotation(i8* %test1, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.1, i32 0, i32 0), i32 10)
  %0 = bitcast [5 x float]* %test to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %0, i8* bitcast ([5 x float]* @main.test to i8*), i32 20, i32 4, i1 false)
  %arraydecay = getelementptr inbounds [5 x float], [5 x float]* %test, i32 0, i32 0
  store float* %arraydecay, float** %abc.addr.i, align 4
  %abc.addr1.i = bitcast float** %abc.addr.i to i8*
  call void @llvm.var.annotation(i8* %abc.addr1.i, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str.1, i32 0, i32 0), i32 4) #1
  %1 = load float*, float** %abc.addr.i, align 4
  %arrayidx.i = getelementptr inbounds float, float* %1, i32 1
  %2 = load float, float* %arrayidx.i, align 4
  %add.i = fadd float %2, 5.000000e+00
  store float %add.i, float* %arrayidx.i, align 4
  %arrayidx = getelementptr inbounds [5 x float], [5 x float]* %test, i32 0, i32 0
  %3 = load float, float* %arrayidx, align 4
  %conv = fpext float %3 to double
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str.2, i32 0, i32 0), double %conv)
  ret i32 0
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i32, i1) #3

declare i32 @printf(i8*, ...) #4

attributes #0 = { alwaysinline nounwind ssp "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { noinline nounwind ssp "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind }
attributes #4 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 4.0.0 (tags/RELEASE_400/final)"}
