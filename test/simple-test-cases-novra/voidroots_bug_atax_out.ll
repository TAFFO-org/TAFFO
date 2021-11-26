;;;TAFFO_TEST_ARGS -disable-vra
; ModuleID = 'linear-algebra/kernels/atax/atax.c'
source_filename = "linear-algebra/kernels/atax/atax.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.13.0"

%struct.__sFILE = type { i8*, i32, i32, i16, i16, %struct.__sbuf, i32, i8*, i32 (i8*)*, i32 (i8*, i8*, i32)*, i64 (i8*, i64, i32)*, i32 (i8*, i8*, i32)*, %struct.__sbuf, %struct.__sFILEX*, i32, [3 x i8], [1 x i8], %struct.__sbuf, i32, i64 }
%struct.__sFILEX = type opaque
%struct.__sbuf = type { i8*, i32 }

@.str = private unnamed_addr constant [19 x i8] c"range -32768 32768\00", section "llvm.metadata"
@.str.1 = private unnamed_addr constant [35 x i8] c"linear-algebra/kernels/atax/atax.c\00", section "llvm.metadata"
@__stderrp = external global %struct.__sFILE*, align 8
@.str.2 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1
@.str.3 = private unnamed_addr constant [9 x i8] c"%0.16lf \00", align 1

; Function Attrs: noinline nounwind ssp uwtable
define i32 @main(i32 %argc, i8** %argv) #0 {
entry:
  %m.addr.i21 = alloca i32, align 4
  %n.addr.i22 = alloca i32, align 4
  %A.addr.i23 = alloca [410 x double]*, align 8
  %x.addr.i24 = alloca double*, align 8
  %y.addr.i25 = alloca double*, align 8
  %tmp.addr.i = alloca double*, align 8
  %i.i26 = alloca i32, align 4
  %j.i27 = alloca i32, align 4
  %n.addr.i11 = alloca i32, align 4
  %y.addr.i = alloca double*, align 8
  %i.i12 = alloca i32, align 4
  %m.addr.i = alloca i32, align 4
  %n.addr.i = alloca i32, align 4
  %A.addr.i = alloca [410 x double]*, align 8
  %x.addr.i = alloca double*, align 8
  %i.i = alloca i32, align 4
  %j.i = alloca i32, align 4
  %fn.i = alloca double, align 8
  %retval = alloca i32, align 4
  %argc.addr = alloca i32, align 4
  %argv.addr = alloca i8**, align 8
  %m = alloca i32, align 4
  %n = alloca i32, align 4
  %A = alloca [390 x [410 x double]], align 16
  %x = alloca [410 x double], align 16
  %y = alloca [410 x double], align 16
  %tmp = alloca [390 x double], align 16
  store i32 0, i32* %retval, align 4
  store i32 %argc, i32* %argc.addr, align 4
  store i8** %argv, i8*** %argv.addr, align 8
  store i32 390, i32* %m, align 4
  store i32 410, i32* %n, align 4
  %A1 = bitcast [390 x [410 x double]]* %A to i8*
  call void @llvm.var.annotation(i8* %A1, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.1, i32 0, i32 0), i32 96)
  %x2 = bitcast [410 x double]* %x to i8*
  call void @llvm.var.annotation(i8* %x2, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.1, i32 0, i32 0), i32 97)
  %y3 = bitcast [410 x double]* %y to i8*
  call void @llvm.var.annotation(i8* %y3, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.1, i32 0, i32 0), i32 98)
  %tmp4 = bitcast [390 x double]* %tmp to i8*
  call void @llvm.var.annotation(i8* %tmp4, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.1, i32 0, i32 0), i32 99)
  %0 = load i32, i32* %m, align 4
  %1 = load i32, i32* %n, align 4
  %arraydecay = getelementptr inbounds [390 x [410 x double]], [390 x [410 x double]]* %A, i32 0, i32 0
  %arraydecay5 = getelementptr inbounds [410 x double], [410 x double]* %x, i32 0, i32 0
  store i32 %0, i32* %m.addr.i, align 4
  store i32 %1, i32* %n.addr.i, align 4
  store [410 x double]* %arraydecay, [410 x double]** %A.addr.i, align 8
  %A.addr1.i = bitcast [410 x double]** %A.addr.i to i8*
  call void @llvm.var.annotation(i8* %A.addr1.i, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.1, i32 0, i32 0), i32 27) #1
  store double* %arraydecay5, double** %x.addr.i, align 8
  %x.addr2.i = bitcast double** %x.addr.i to i8*
  call void @llvm.var.annotation(i8* %x.addr2.i, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.1, i32 0, i32 0), i32 28) #1
  %fn3.i = bitcast double* %fn.i to i8*
  call void @llvm.var.annotation(i8* %fn3.i, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.1, i32 0, i32 0), i32 31) #1
  %2 = load i32, i32* %n.addr.i, align 4
  %conv.i = sitofp i32 %2 to double
  store double %conv.i, double* %fn.i, align 8
  store i32 0, i32* %i.i, align 4
  br label %for.cond.i

for.cond.i:                                       ; preds = %for.body.i, %entry
  %3 = load i32, i32* %i.i, align 4
  %4 = load i32, i32* %n.addr.i, align 4
  %cmp.i = icmp slt i32 %3, %4
  br i1 %cmp.i, label %for.body.i, label %for.end.i

for.body.i:                                       ; preds = %for.cond.i
  %5 = load i32, i32* %i.i, align 4
  %conv5.i = sitofp i32 %5 to double
  %6 = load double, double* %fn.i, align 8
  %div.i = fdiv double %conv5.i, %6
  %add.i = fadd double 1.000000e+00, %div.i
  %7 = load double*, double** %x.addr.i, align 8
  %8 = load i32, i32* %i.i, align 4
  %idxprom.i = sext i32 %8 to i64
  %arrayidx.i = getelementptr inbounds double, double* %7, i64 %idxprom.i
  store double %add.i, double* %arrayidx.i, align 8
  %9 = load i32, i32* %i.i, align 4
  %inc.i = add nsw i32 %9, 1
  store i32 %inc.i, i32* %i.i, align 4
  br label %for.cond.i

for.end.i:                                        ; preds = %for.cond.i
  store i32 0, i32* %i.i, align 4
  br label %for.cond6.i

for.cond6.i:                                      ; preds = %for.end24.i, %for.end.i
  %10 = load i32, i32* %i.i, align 4
  %11 = load i32, i32* %m.addr.i, align 4
  %cmp7.i = icmp slt i32 %10, %11
  br i1 %cmp7.i, label %for.body9.i, label %init_array.exit

for.body9.i:                                      ; preds = %for.cond6.i
  store i32 0, i32* %j.i, align 4
  br label %for.cond10.i

for.cond10.i:                                     ; preds = %for.body13.i, %for.body9.i
  %12 = load i32, i32* %j.i, align 4
  %13 = load i32, i32* %n.addr.i, align 4
  %cmp11.i = icmp slt i32 %12, %13
  br i1 %cmp11.i, label %for.body13.i, label %for.end24.i

for.body13.i:                                     ; preds = %for.cond10.i
  %14 = load i32, i32* %i.i, align 4
  %15 = load i32, i32* %j.i, align 4
  %add14.i = add nsw i32 %14, %15
  %16 = load i32, i32* %n.addr.i, align 4
  %rem.i = srem i32 %add14.i, %16
  %conv15.i = sitofp i32 %rem.i to double
  %17 = load i32, i32* %m.addr.i, align 4
  %mul.i = mul nsw i32 5, %17
  %conv16.i = sitofp i32 %mul.i to double
  %div17.i = fdiv double %conv15.i, %conv16.i
  %18 = load [410 x double]*, [410 x double]** %A.addr.i, align 8
  %19 = load i32, i32* %i.i, align 4
  %idxprom18.i = sext i32 %19 to i64
  %arrayidx19.i = getelementptr inbounds [410 x double], [410 x double]* %18, i64 %idxprom18.i
  %20 = load i32, i32* %j.i, align 4
  %idxprom20.i = sext i32 %20 to i64
  %arrayidx21.i = getelementptr inbounds [410 x double], [410 x double]* %arrayidx19.i, i64 0, i64 %idxprom20.i
  store double %div17.i, double* %arrayidx21.i, align 8
  %21 = load i32, i32* %j.i, align 4
  %inc23.i = add nsw i32 %21, 1
  store i32 %inc23.i, i32* %j.i, align 4
  br label %for.cond10.i

for.end24.i:                                      ; preds = %for.cond10.i
  %22 = load i32, i32* %i.i, align 4
  %inc26.i = add nsw i32 %22, 1
  store i32 %inc26.i, i32* %i.i, align 4
  br label %for.cond6.i

init_array.exit:                                  ; preds = %for.cond6.i
  call void (...) @polybench_timer_start()
  %23 = load i32, i32* %m, align 4
  %24 = load i32, i32* %n, align 4
  %arraydecay6 = getelementptr inbounds [390 x [410 x double]], [390 x [410 x double]]* %A, i32 0, i32 0
  %arraydecay7 = getelementptr inbounds [410 x double], [410 x double]* %x, i32 0, i32 0
  %arraydecay8 = getelementptr inbounds [410 x double], [410 x double]* %y, i32 0, i32 0
  %arraydecay9 = getelementptr inbounds [390 x double], [390 x double]* %tmp, i32 0, i32 0
  store i32 %23, i32* %m.addr.i21, align 4
  store i32 %24, i32* %n.addr.i22, align 4
  store [410 x double]* %arraydecay6, [410 x double]** %A.addr.i23, align 8
  %A.addr1.i28 = bitcast [410 x double]** %A.addr.i23 to i8*
  call void @llvm.var.annotation(i8* %A.addr1.i28, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.1, i32 0, i32 0), i32 66) #1
  store double* %arraydecay7, double** %x.addr.i24, align 8
  %x.addr2.i29 = bitcast double** %x.addr.i24 to i8*
  call void @llvm.var.annotation(i8* %x.addr2.i29, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.1, i32 0, i32 0), i32 67) #1
  store double* %arraydecay8, double** %y.addr.i25, align 8
  %y.addr3.i = bitcast double** %y.addr.i25 to i8*
  call void @llvm.var.annotation(i8* %y.addr3.i, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.1, i32 0, i32 0), i32 68) #1
  store double* %arraydecay9, double** %tmp.addr.i, align 8
  %tmp.addr4.i = bitcast double** %tmp.addr.i to i8*
  call void @llvm.var.annotation(i8* %tmp.addr4.i, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.1, i32 0, i32 0), i32 69) #1
  store i32 0, i32* %i.i26, align 4
  br label %for.cond.i31

for.cond.i31:                                     ; preds = %for.body.i34, %init_array.exit
  %25 = load i32, i32* %i.i26, align 4
  %26 = load i32, i32* %n.addr.i22, align 4
  %cmp.i30 = icmp slt i32 %25, %26
  br i1 %cmp.i30, label %for.body.i34, label %for.end.i36

for.body.i34:                                     ; preds = %for.cond.i31
  %27 = load double*, double** %y.addr.i25, align 8
  %28 = load i32, i32* %i.i26, align 4
  %idxprom.i32 = sext i32 %28 to i64
  %arrayidx.i33 = getelementptr inbounds double, double* %27, i64 %idxprom.i32
  store double 0.000000e+00, double* %arrayidx.i33, align 8
  %29 = load i32, i32* %i.i26, align 4
  %inc.i35 = add nsw i32 %29, 1
  store i32 %inc.i35, i32* %i.i26, align 4
  br label %for.cond.i31

for.end.i36:                                      ; preds = %for.cond.i31
  store i32 0, i32* %i.i26, align 4
  br label %for.cond7.i

for.cond7.i:                                      ; preds = %for.end45.i, %for.end.i36
  %30 = load i32, i32* %i.i26, align 4
  %31 = load i32, i32* %m.addr.i21, align 4
  %cmp8.i = icmp slt i32 %30, %31
  br i1 %cmp8.i, label %for.body9.i37, label %kernel_atax.exit

for.body9.i37:                                    ; preds = %for.cond7.i
  %32 = load double*, double** %tmp.addr.i, align 8
  %33 = load i32, i32* %i.i26, align 4
  %idxprom10.i = sext i32 %33 to i64
  %arrayidx11.i = getelementptr inbounds double, double* %32, i64 %idxprom10.i
  store double 0.000000e+00, double* %arrayidx11.i, align 8
  store i32 0, i32* %j.i27, align 4
  br label %for.cond12.i

for.cond12.i:                                     ; preds = %for.body14.i, %for.body9.i37
  %34 = load i32, i32* %j.i27, align 4
  %35 = load i32, i32* %n.addr.i22, align 4
  %cmp13.i = icmp slt i32 %34, %35
  br i1 %cmp13.i, label %for.body14.i, label %for.end27.i

for.body14.i:                                     ; preds = %for.cond12.i
  %36 = load double*, double** %tmp.addr.i, align 8
  %37 = load i32, i32* %i.i26, align 4
  %idxprom15.i = sext i32 %37 to i64
  %arrayidx16.i = getelementptr inbounds double, double* %36, i64 %idxprom15.i
  %38 = load double, double* %arrayidx16.i, align 8
  %39 = load [410 x double]*, [410 x double]** %A.addr.i23, align 8
  %40 = load i32, i32* %i.i26, align 4
  %idxprom17.i = sext i32 %40 to i64
  %arrayidx18.i = getelementptr inbounds [410 x double], [410 x double]* %39, i64 %idxprom17.i
  %41 = load i32, i32* %j.i27, align 4
  %idxprom19.i = sext i32 %41 to i64
  %arrayidx20.i = getelementptr inbounds [410 x double], [410 x double]* %arrayidx18.i, i64 0, i64 %idxprom19.i
  %42 = load double, double* %arrayidx20.i, align 8
  %43 = load double*, double** %x.addr.i24, align 8
  %44 = load i32, i32* %j.i27, align 4
  %idxprom21.i = sext i32 %44 to i64
  %arrayidx22.i = getelementptr inbounds double, double* %43, i64 %idxprom21.i
  %45 = load double, double* %arrayidx22.i, align 8
  %mul.i38 = fmul double %42, %45
  %add.i39 = fadd double %38, %mul.i38
  %46 = load double*, double** %tmp.addr.i, align 8
  %47 = load i32, i32* %i.i26, align 4
  %idxprom23.i = sext i32 %47 to i64
  %arrayidx24.i = getelementptr inbounds double, double* %46, i64 %idxprom23.i
  store double %add.i39, double* %arrayidx24.i, align 8
  %48 = load i32, i32* %j.i27, align 4
  %inc26.i40 = add nsw i32 %48, 1
  store i32 %inc26.i40, i32* %j.i27, align 4
  br label %for.cond12.i

for.end27.i:                                      ; preds = %for.cond12.i
  store i32 0, i32* %j.i27, align 4
  br label %for.cond28.i

for.cond28.i:                                     ; preds = %for.body30.i, %for.end27.i
  %49 = load i32, i32* %j.i27, align 4
  %50 = load i32, i32* %n.addr.i22, align 4
  %cmp29.i = icmp slt i32 %49, %50
  br i1 %cmp29.i, label %for.body30.i, label %for.end45.i

for.body30.i:                                     ; preds = %for.cond28.i
  %51 = load double*, double** %y.addr.i25, align 8
  %52 = load i32, i32* %j.i27, align 4
  %idxprom31.i = sext i32 %52 to i64
  %arrayidx32.i = getelementptr inbounds double, double* %51, i64 %idxprom31.i
  %53 = load double, double* %arrayidx32.i, align 8
  %54 = load [410 x double]*, [410 x double]** %A.addr.i23, align 8
  %55 = load i32, i32* %i.i26, align 4
  %idxprom33.i = sext i32 %55 to i64
  %arrayidx34.i = getelementptr inbounds [410 x double], [410 x double]* %54, i64 %idxprom33.i
  %56 = load i32, i32* %j.i27, align 4
  %idxprom35.i = sext i32 %56 to i64
  %arrayidx36.i = getelementptr inbounds [410 x double], [410 x double]* %arrayidx34.i, i64 0, i64 %idxprom35.i
  %57 = load double, double* %arrayidx36.i, align 8
  %58 = load double*, double** %tmp.addr.i, align 8
  %59 = load i32, i32* %i.i26, align 4
  %idxprom37.i = sext i32 %59 to i64
  %arrayidx38.i = getelementptr inbounds double, double* %58, i64 %idxprom37.i
  %60 = load double, double* %arrayidx38.i, align 8
  %mul39.i = fmul double %57, %60
  %add40.i = fadd double %53, %mul39.i
  %61 = load double*, double** %y.addr.i25, align 8
  %62 = load i32, i32* %j.i27, align 4
  %idxprom41.i = sext i32 %62 to i64
  %arrayidx42.i = getelementptr inbounds double, double* %61, i64 %idxprom41.i
  store double %add40.i, double* %arrayidx42.i, align 8
  %63 = load i32, i32* %j.i27, align 4
  %inc44.i = add nsw i32 %63, 1
  store i32 %inc44.i, i32* %j.i27, align 4
  br label %for.cond28.i

for.end45.i:                                      ; preds = %for.cond28.i
  %64 = load i32, i32* %i.i26, align 4
  %inc47.i = add nsw i32 %64, 1
  store i32 %inc47.i, i32* %i.i26, align 4
  br label %for.cond7.i

kernel_atax.exit:                                 ; preds = %for.cond7.i
  call void (...) @polybench_timer_stop()
  call void (...) @polybench_timer_print()
  %65 = load i32, i32* %n, align 4
  %arraydecay10 = getelementptr inbounds [410 x double], [410 x double]* %y, i32 0, i32 0
  store i32 %65, i32* %n.addr.i11, align 4
  store double* %arraydecay10, double** %y.addr.i, align 8
  %y.addr1.i = bitcast double** %y.addr.i to i8*
  call void @llvm.var.annotation(i8* %y.addr1.i, i8* getelementptr inbounds ([19 x i8], [19 x i8]* @.str, i32 0, i32 0), i8* getelementptr inbounds ([35 x i8], [35 x i8]* @.str.1, i32 0, i32 0), i32 46) #1
  store i32 0, i32* %i.i12, align 4
  br label %for.cond.i14

for.cond.i14:                                     ; preds = %if.end.i, %kernel_atax.exit
  %66 = load i32, i32* %i.i12, align 4
  %67 = load i32, i32* %n.addr.i11, align 4
  %cmp.i13 = icmp slt i32 %66, %67
  br i1 %cmp.i13, label %for.body.i16, label %print_array.exit

for.body.i16:                                     ; preds = %for.cond.i14
  %68 = load i32, i32* %i.i12, align 4
  %rem.i15 = srem i32 %68, 20
  %cmp2.i = icmp eq i32 %rem.i15, 0
  br i1 %cmp2.i, label %if.then.i, label %if.end.i

if.then.i:                                        ; preds = %for.body.i16
  %69 = load %struct.__sFILE*, %struct.__sFILE** @__stderrp, align 8
  %call.i = call i32 (%struct.__sFILE*, i8*, ...) @fprintf(%struct.__sFILE* %69, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.2, i32 0, i32 0)) #1
  br label %if.end.i

if.end.i:                                         ; preds = %if.then.i, %for.body.i16
  %70 = load %struct.__sFILE*, %struct.__sFILE** @__stderrp, align 8
  %71 = load double*, double** %y.addr.i, align 8
  %72 = load i32, i32* %i.i12, align 4
  %idxprom.i17 = sext i32 %72 to i64
  %arrayidx.i18 = getelementptr inbounds double, double* %71, i64 %idxprom.i17
  %73 = load double, double* %arrayidx.i18, align 8
  %call3.i = call i32 (%struct.__sFILE*, i8*, ...) @fprintf(%struct.__sFILE* %70, i8* getelementptr inbounds ([9 x i8], [9 x i8]* @.str.3, i32 0, i32 0), double %73) #1
  %74 = load i32, i32* %i.i12, align 4
  %inc.i19 = add nsw i32 %74, 1
  store i32 %inc.i19, i32* %i.i12, align 4
  br label %for.cond.i14

print_array.exit:                                 ; preds = %for.cond.i14
  ret i32 0
}

; Function Attrs: nounwind
declare void @llvm.var.annotation(i8*, i8*, i8*, i32) #1

declare void @polybench_timer_start(...) #2

declare void @polybench_timer_stop(...) #2

declare void @polybench_timer_print(...) #2

declare i32 @fprintf(%struct.__sFILE*, i8*, ...) #2

attributes #0 = { noinline nounwind ssp uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="penryn" "target-features"="+cx16,+fxsr,+mmx,+sse,+sse2,+sse3,+sse4.1,+ssse3,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"PIC Level", i32 2}
!1 = !{!"clang version 4.0.0 (tags/RELEASE_400/final)"}
