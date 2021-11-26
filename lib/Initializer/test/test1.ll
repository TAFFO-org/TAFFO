; ModuleID = 'test1.c'
source_filename = "test1.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

@.str = private unnamed_addr constant [9 x i8] c"no_float\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [8 x i8] c"test1.c\00", section "llvm.metadata"
@global = common global float 0.000000e+00, align 4

; Function Attrs: noinline nounwind ssp uwtable
define float @test(float %param, i32 %notafloat) #0 {
entry:
  %param.addr = alloca float, align 4
  %notafloat.addr = alloca i32, align 4
  %notafloat2 = alloca i32, align 4
  %local = alloca float, align 4
  store float %param, float* %param.addr, align 4
  store i32 %notafloat, i32* %notafloat.addr, align 4
  %local1 = bitcast float* %local to i8*
  call void @llvm.var.annotation(i8* %local1, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.1, i32 0, i32 0), i32 8)
  store float 2.000000e+00, float* %local, align 4
  %0 = load float, float* %param.addr, align 4
  %1 = load float, float* %local, align 4
  %mul = fmul float %1, %0
  store float %mul, float* %local, align 4
  %2 = load i32, i32* %notafloat.addr, align 4
  %conv = sitofp i32 %2 to float
  %3 = load float, float* %local, align 4
  %add = fadd float %3, %conv
  store float %add, float* %local, align 4
  %4 = load float, float* %local, align 4
  %conv2 = fptosi float %4 to i32
  store i32 %conv2, i32* %notafloat2, align 4
  %5 = load i32, i32* %notafloat2, align 4
  %conv3 = sitofp i32 %5 to float
  ret float %conv3
}

; Function Attrs: nounwind
declare void @llvm.var.annotation(i8*, i8*, i8*, i32) #1

; Function Attrs: noinline nounwind ssp uwtable
define i32 @test2(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  %0 = load i32, i32* %a.addr, align 4
  %conv = sitofp i32 %0 to double
  %add = fadd double %conv, 2.000000e+00
  %conv1 = fptosi double %add to i32
  ret i32 %conv1
}

attributes #0 = { noinline nounwind ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 4.0.0 (tags/RELEASE_400/final)"}
