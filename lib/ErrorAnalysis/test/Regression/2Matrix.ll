; RUN:  opt -load %errorproplib -globals-aa -cfl-steens-aa -cfl-anders-aa -tbaa -errorprop -nounroll -S %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"


define void @foo(i32 %alpha, i32 %beta, [22 x i32]* %A, [18 x i32]* noalias nocapture %B, [24 x i32]* noalias nocapture %C, [24 x i32]* noalias nocapture %D) !taffo.funinfo !0 {
entry:
  %tmp = alloca [16 x [18 x i32]], align 16, !taffo.info !14
  br label %for.cond

for.cond:                                         ; preds = %for.inc25, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc26, %for.inc25 ]
  %cmp = icmp slt i32 %i.0, 16
  br i1 %cmp, label %for.body, label %for.end27

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc22, %for.body
  %j.0 = phi i32 [ 0, %for.body ], [ %inc23, %for.inc22 ]
  %cmp2 = icmp slt i32 %j.0, 18
  br i1 %cmp2, label %for.body3, label %for.end24

for.body3:                                        ; preds = %for.cond1
  %idxprom = sext i32 %i.0 to i64
  %arrayidx = getelementptr inbounds [16 x [18 x i32]], [16 x [18 x i32]]* %tmp, i64 0, i64 %idxprom
  %idxprom4 = sext i32 %j.0 to i64
  %arrayidx5 = getelementptr inbounds [18 x i32], [18 x i32]* %arrayidx, i64 0, i64 %idxprom4
  store i32 0, i32* %arrayidx5, align 4, !taffo.info !17
  br label %for.cond6

for.cond6:                                        ; preds = %for.inc, %for.body3
  %k.0 = phi i32 [ 0, %for.body3 ], [ %inc, %for.inc ]
  %cmp7 = icmp slt i32 %k.0, 22
  br i1 %cmp7, label %for.body8, label %for.end

for.body8:                                        ; preds = %for.cond6
  %idxprom9 = sext i32 %i.0 to i64
  %arrayidx10 = getelementptr inbounds [22 x i32], [22 x i32]* %A, i64 %idxprom9
  %idxprom11 = sext i32 %k.0 to i64
  %arrayidx12 = getelementptr inbounds [22 x i32], [22 x i32]* %arrayidx10, i64 0, i64 %idxprom11
  %0 = load i32, i32* %arrayidx12, align 4, !taffo.info !19
  %mul = mul nsw i32 %alpha, %0, !taffo.info !20
  %idxprom13 = sext i32 %k.0 to i64
  %arrayidx14 = getelementptr inbounds [18 x i32], [18 x i32]* %B, i64 %idxprom13
  %idxprom15 = sext i32 %j.0 to i64
  %arrayidx16 = getelementptr inbounds [18 x i32], [18 x i32]* %arrayidx14, i64 0, i64 %idxprom15
  %1 = load i32, i32* %arrayidx16, align 4, !taffo.info !22
  %mul17 = mul nsw i32 %mul, %1, !taffo.info !23
  %idxprom18 = sext i32 %i.0 to i64
  %arrayidx19 = getelementptr inbounds [16 x [18 x i32]], [16 x [18 x i32]]* %tmp, i64 0, i64 %idxprom18
  %idxprom20 = sext i32 %j.0 to i64
  %arrayidx21 = getelementptr inbounds [18 x i32], [18 x i32]* %arrayidx19, i64 0, i64 %idxprom20
  %2 = load i32, i32* %arrayidx21, align 4, !taffo.info !17
  %add = add nsw i32 %2, %mul17, !taffo.info !14
  store i32 %add, i32* %arrayidx21, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body8
  %inc = add nsw i32 %k.0, 1
  br label %for.cond6

for.end:                                          ; preds = %for.cond6
  br label %for.inc22

for.inc22:                                        ; preds = %for.end
  %inc23 = add nsw i32 %j.0, 1
  br label %for.cond1

for.end24:                                        ; preds = %for.cond1
  br label %for.inc25

for.inc25:                                        ; preds = %for.end24
  %inc26 = add nsw i32 %i.0, 1
  br label %for.cond

for.end27:                                        ; preds = %for.cond
  br label %for.cond28

for.cond28:                                       ; preds = %for.inc62, %for.end27
  %i.1 = phi i32 [ 0, %for.end27 ], [ %inc63, %for.inc62 ]
  %cmp29 = icmp slt i32 %i.1, 16
  br i1 %cmp29, label %for.body30, label %for.end64

for.body30:                                       ; preds = %for.cond28
  br label %for.cond31

for.cond31:                                       ; preds = %for.inc59, %for.body30
  %j.1 = phi i32 [ 0, %for.body30 ], [ %inc60, %for.inc59 ]
  %cmp32 = icmp slt i32 %j.1, 24
  br i1 %cmp32, label %for.body33, label %for.end61

for.body33:                                       ; preds = %for.cond31
  %idxprom34 = sext i32 %i.1 to i64
  %arrayidx35 = getelementptr inbounds [24 x i32], [24 x i32]* %D, i64 %idxprom34
  %idxprom36 = sext i32 %j.1 to i64
  %arrayidx37 = getelementptr inbounds [24 x i32], [24 x i32]* %arrayidx35, i64 0, i64 %idxprom36
  %3 = load i32, i32* %arrayidx37, align 4, !taffo.info !19
  %mul38 = mul nsw i32 %3, %beta, !taffo.info !25
  store i32 %mul38, i32* %arrayidx37, align 4
  br label %for.cond39

for.cond39:                                       ; preds = %for.inc56, %for.body33
  %k.1 = phi i32 [ 0, %for.body33 ], [ %inc57, %for.inc56 ]
  %cmp40 = icmp slt i32 %k.1, 18
  br i1 %cmp40, label %for.body41, label %for.end58

for.body41:                                       ; preds = %for.cond39
  %idxprom42 = sext i32 %i.1 to i64
  %arrayidx43 = getelementptr inbounds [16 x [18 x i32]], [16 x [18 x i32]]* %tmp, i64 0, i64 %idxprom42
  %idxprom44 = sext i32 %k.1 to i64
  %arrayidx45 = getelementptr inbounds [18 x i32], [18 x i32]* %arrayidx43, i64 0, i64 %idxprom44
  %4 = load i32, i32* %arrayidx45, align 4, !taffo.info !14
  %idxprom46 = sext i32 %k.1 to i64
  %arrayidx47 = getelementptr inbounds [24 x i32], [24 x i32]* %C, i64 %idxprom46
  %idxprom48 = sext i32 %j.1 to i64
  %arrayidx49 = getelementptr inbounds [24 x i32], [24 x i32]* %arrayidx47, i64 0, i64 %idxprom48
  %5 = load i32, i32* %arrayidx49, align 4, !taffo.info !27
  %mul50 = mul nsw i32 %4, %5, !taffo.info !28
  %idxprom51 = sext i32 %i.1 to i64
  %arrayidx52 = getelementptr inbounds [24 x i32], [24 x i32]* %D, i64 %idxprom51
  %idxprom53 = sext i32 %j.1 to i64
  %arrayidx54 = getelementptr inbounds [24 x i32], [24 x i32]* %arrayidx52, i64 0, i64 %idxprom53
  %6 = load i32, i32* %arrayidx54, align 4, !taffo.info !30
  %add55 = add nsw i32 %6, %mul50, !taffo.info !30
  store i32 %add55, i32* %arrayidx54, align 4
  br label %for.inc56

for.inc56:                                        ; preds = %for.body41
  %inc57 = add nsw i32 %k.1, 1
  br label %for.cond39

for.end58:                                        ; preds = %for.cond39
  br label %for.inc59

for.inc59:                                        ; preds = %for.end58
  %inc60 = add nsw i32 %j.1, 1
  br label %for.cond31

for.end61:                                        ; preds = %for.cond31
  br label %for.inc62

for.inc62:                                        ; preds = %for.end61
  %inc63 = add nsw i32 %i.1, 1
  br label %for.cond28

for.end64:                                        ; preds = %for.cond28
  ret void
}

!0 = !{i32 1, !1, i32 1, !2, i32 1, !3, i32 1, !4, i32 1, !5, i32 1, !6}
!1 = !{!7, !8, !9}
!7 = !{!"fixp", i32 32, i32 17}
!8 = !{double 5.000000e+00, double 5.000000e+00}
!9 = !{double 2.500000e-05}
!2 = !{!7, !10, !9}
!10 = !{double 2.000000e+00, double 3.000000e+00}
!3 = !{!7, !11, !9}
!11 = !{double 1.000000e+00, double 3.000000e+01}
!4 = !{!7, !12, !9}
!12 = !{double 1.000000e+00, double 4.000000e+01}
!5 = !{!7, !13, !9}
!13 = !{double 1.000000e+00, double 2.500000e+01}
!6 = !{!7, !11, !9}
!14 = !{!7, !15, i1 0}
!15 = !{double 1.000000e+01, double 1.200000e+04}
!17 = !{!7, !18, i1 0}
!18 = !{double 0.000000e+00, double 0.000000e+00}
!19 = !{!7, !11, i1 0}
!20 = !{!7, !21, i1 0}
!21 = !{double 5.000000e+00, double 1.500000e+02}
!22 = !{!7, !12, i1 0}
!23 = !{!7, !24, i1 0}
!24 = !{double 5.000000e+00, double 6.000000e+03}
!25 = !{!7, !26, i1 0}
!26 = !{double 2.000000e+00, double 7.500000e+01}
!27 = !{!7, !13, i1 0}
!28 = !{!7, !29, i1 0}
!29 = !{double 1.000000e+01, double 3.276800e+04}
!30 = !{!7, !31, i1 0}
!31 = !{double 1.200000e+02, double 3.276800e+04}

; CHECK-DAG: !{double 0x3EE0000000000000}
; CHECK-DAG: !{double 0x3F4CAC0988BFD79C}
; CHECK-DAG: !{double 0x3FA3D70BD017E3B4}
; CHECK-DAG: !{double 0x3FA3D80BD017E3B4}
; CHECK-DAG: !{double 0x3FBDF548E32F8F2E}
; CHECK-DAG: !{double 0x3FF44D970B78A4FC}
; CHECK-DAG: !{double 0x3FF62CEB99AB9DEF}

;  store i32 0, i32* %arrayidx5, align 4, !taffo.info !15, !taffo.abserror !17
;  %0 = load i32, i32* %arrayidx12, align 4, !taffo.info !18, !taffo.abserror !4
;  %mul = mul nsw i32 %alpha, %0, !taffo.info !19, !taffo.abserror !21
;  %1 = load i32, i32* %arrayidx16, align 4, !taffo.info !22, !taffo.abserror !4
;  %mul17 = mul nsw i32 %mul, %1, !taffo.info !23, !taffo.abserror !25
;  %2 = load i32, i32* %arrayidx21, align 4, !taffo.info !15, !taffo.abserror !17
;  %add = add nsw i32 %2, %mul17, !taffo.info !13, !taffo.abserror !26
;  store i32 %add, i32* %arrayidx21, align 4, !taffo.abserror !26
;  %3 = load i32, i32* %arrayidx37, align 4, !taffo.info !18, !taffo.abserror !26
;  %mul38 = mul nsw i32 %3, %beta, !taffo.info !27, !taffo.abserror !29
;  store i32 %mul38, i32* %arrayidx37, align 4, !taffo.abserror !29
;  %4 = load i32, i32* %arrayidx45, align 4, !taffo.info !13, !taffo.abserror !26
;  %5 = load i32, i32* %arrayidx49, align 4, !taffo.info !30, !taffo.abserror !4
;  %mul50 = mul nsw i32 %4, %5, !taffo.info !31, !taffo.abserror !33
;  %6 = load i32, i32* %arrayidx54, align 4, !taffo.info !34, !taffo.abserror !29
;  %add55 = add nsw i32 %6, %mul50, !taffo.info !34, !taffo.abserror !36
;  store i32 %add55, i32* %arrayidx54, align 4, !taffo.abserror !36
;  !17 = !{double 0x3EE0000000000000}
;  !21 = !{double 0x3F4CAC0988BFD79C}
;  !25 = !{double 0x3FA3D70BD017E3B4}
;  !26 = !{double 0x3FA3D80BD017E3B4}
;  !29 = !{double 0x3FBDF548E32F8F2E}
;  !33 = !{double 0x3FF44D970B78A4FC}
;  !36 = !{double 0x3FF62CEB99AB9DEF}
