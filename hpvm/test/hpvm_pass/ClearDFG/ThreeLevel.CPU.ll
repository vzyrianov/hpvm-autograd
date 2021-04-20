; RUN: opt -load LLVMBuildDFG.so -load LLVMDFG2LLVM_CPU.so -load LLVMClearDFG.so -S -dfg2llvm-cpu -clearDFG <  %s | FileCheck %s
; ModuleID = 'ThreeLevel.ll'
source_filename = "ThreeLevel.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.Root = type { i32*, i64, i32*, i64, i32*, i64 }
%struct.out.Func1 = type <{ i32* }>
%struct.out.Func3 = type <{ i32* }>
%struct.out.Func2 = type <{ i32* }>
%struct.out.PipeRoot = type <{ i32* }>


; CHECK-LABEL: i32 @main(
; CHECK-NOT: call void @llvm.hpvm.init()
; CHECK: call i8* @llvm_hpvm_cpu_launch(i8* (i8*)* @LaunchDataflowGraph, i8*
; CHECK-NOT: call i8* @llvm.hpvm.launch(i8*
; CHECK: call void @llvm_hpvm_cpu_wait(i8*

; CHECK-LABEL: @Func1_cloned.1_cloned_cloned_cloned_cloned_cloned_cloned
; CHECK: call i8* @llvm_hpvm_cpu_argument_ptr(

; CHECK-LABEL: @Func3_cloned.2_cloned_cloned_cloned_cloned_cloned_cloned(
; CHECK-LABEL: for.body1:
; CHECK: %index.y = phi i64 [ 0, %for.body ], [ %index.y.inc, %for.body1 ]
; CHECK-NEXT: call void @llvm_hpvm_cpu_dstack_push(
; CHECK-NEXT: @Func1_cloned.1_cloned_cloned_cloned_cloned_cloned_cloned(
; CHECK-NEXT: call void @llvm_hpvm_cpu_dstack_pop()

; CHECK-LABEL: @Func2_cloned.3_cloned_cloned_cloned_cloned_cloned_cloned(
; CHECK-LABEL: for.body:
; CHECK-NEXT: %index.x = phi i64 [ 0, %entry ], [ %index.x.inc, %for.body ]
; CHECK-NEXT: call void @llvm_hpvm_cpu_dstack_push(
; CHECK-NEXT: @Func3_cloned.2_cloned_cloned_cloned_cloned_cloned_cloned(
; CHECK-NEXT: call void @llvm_hpvm_cpu_dstack_pop()

; CHECK-LABEL: @PipeRoot_cloned.4(
; CHECK: call void @llvm_hpvm_cpu_dstack_push(
; CHECK-NEXT: @Func2_cloned.3_cloned_cloned_cloned_cloned_cloned_cloned(
; CHECK-NEXT: call void @llvm_hpvm_cpu_dstack_pop()

; CHECK-LABEL: @LaunchDataflowGraph(
; CHECK: call %struct.out.PipeRoot @PipeRoot_cloned.4(

declare dso_local void @__hpvm__hint(i32) local_unnamed_addr #0

declare dso_local void @__hpvm__attributes(i32, ...) local_unnamed_addr #0

declare dso_local void @__hpvm__return(i32, ...) local_unnamed_addr #0

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

declare dso_local i8* @__hpvm__createNodeND(i32, ...) local_unnamed_addr #0

declare dso_local void @__hpvm__bindIn(i8*, i32, i32, i32) local_unnamed_addr #0

declare dso_local void @__hpvm__bindOut(i8*, i32, i32, i32) local_unnamed_addr #0

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #2 {
entry:
  %In1 = alloca i32, align 4
  %In2 = alloca i32, align 4
  %Out = alloca i32, align 4
  %RootArgs = alloca %struct.Root, align 8
  %0 = bitcast i32* %In1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %0) #3
  store i32 1, i32* %In1, align 4, !tbaa !6
  %1 = bitcast i32* %In2 to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %1) #3
  store i32 2, i32* %In2, align 4, !tbaa !6
  %2 = bitcast i32* %Out to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* nonnull %2) #3
  store i32 0, i32* %Out, align 4, !tbaa !6
  %3 = bitcast %struct.Root* %RootArgs to i8*
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %3) #3
  %input1 = getelementptr inbounds %struct.Root, %struct.Root* %RootArgs, i64 0, i32 0
  store i32* %In1, i32** %input1, align 8, !tbaa !10
  %Insize1 = getelementptr inbounds %struct.Root, %struct.Root* %RootArgs, i64 0, i32 1
  store i64 32, i64* %Insize1, align 8, !tbaa !14
  %input2 = getelementptr inbounds %struct.Root, %struct.Root* %RootArgs, i64 0, i32 2
  store i32* %In2, i32** %input2, align 8, !tbaa !15
  %Insize2 = getelementptr inbounds %struct.Root, %struct.Root* %RootArgs, i64 0, i32 3
  store i64 32, i64* %Insize2, align 8, !tbaa !16
  %output = getelementptr inbounds %struct.Root, %struct.Root* %RootArgs, i64 0, i32 4
  store i32* %Out, i32** %output, align 8, !tbaa !17
  %Outsize = getelementptr inbounds %struct.Root, %struct.Root* %RootArgs, i64 0, i32 5
  store i64 32, i64* %Outsize, align 8, !tbaa !18
  call void @llvm.hpvm.init()
  %4 = bitcast %struct.Root* %RootArgs to i8*
  %graphID = call i8* @llvm.hpvm.launch(i8* bitcast (%struct.out.PipeRoot (i32*, i64, i32*, i64, i32*, i64)* @PipeRoot_cloned to i8*), i8* %4, i1 false)
  call void @llvm.hpvm.wait(i8* %graphID)
  call void @llvm.hpvm.cleanup()
  call void @llvm.lifetime.end.p0i8(i64 48, i8* nonnull %3) #3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %2) #3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %1) #3
  call void @llvm.lifetime.end.p0i8(i64 4, i8* nonnull %0) #3
  ret i32 0
}

declare dso_local void @__hpvm__init(...) local_unnamed_addr #0

declare dso_local i8* @__hpvm__launch(i32, ...) local_unnamed_addr #0

declare dso_local void @__hpvm__wait(i8*) local_unnamed_addr #0

declare dso_local void @__hpvm__cleanup(...) local_unnamed_addr #0

declare i8* @llvm_hpvm_initializeTimerSet()

declare void @llvm_hpvm_switchToTimer(i8**, i32)

declare void @llvm_hpvm_printTimerSet(i8**, i8*)

; Function Attrs: nounwind uwtable
define dso_local %struct.out.Func1 @Func1_cloned(i32* in %In, i64 %Insize, i32* out %Out, i64 %Outsize) #2 {
entry:
  %returnStruct = insertvalue %struct.out.Func1 undef, i32* %Out, 0
  ret %struct.out.Func1 %returnStruct
}

; Function Attrs: nounwind
declare i8* @llvm.hpvm.createNode2D(i8*, i64, i64) #3

; Function Attrs: nounwind
declare void @llvm.hpvm.bind.input(i8*, i32, i32, i1) #3

; Function Attrs: nounwind
declare void @llvm.hpvm.bind.output(i8*, i32, i32, i1) #3

; Function Attrs: nounwind uwtable
define dso_local %struct.out.Func3 @Func3_cloned(i32* in %In, i64 %Insize, i32* out %Out, i64 %Outsize) #2 {
; CHECK-NOT: @Func3_cloned
entry:
  %Func1_cloned.node = call i8* @llvm.hpvm.createNode2D(i8* bitcast (%struct.out.Func1 (i32*, i64, i32*, i64)* @Func1_cloned to i8*), i64 3, i64 5)
  call void @llvm.hpvm.bind.input(i8* %Func1_cloned.node, i32 0, i32 0, i1 false)
  call void @llvm.hpvm.bind.input(i8* %Func1_cloned.node, i32 1, i32 1, i1 false)
  call void @llvm.hpvm.bind.input(i8* %Func1_cloned.node, i32 2, i32 2, i1 false)
  call void @llvm.hpvm.bind.input(i8* %Func1_cloned.node, i32 3, i32 3, i1 false)
  call void @llvm.hpvm.bind.output(i8* %Func1_cloned.node, i32 0, i32 0, i1 false)
  ret %struct.out.Func3 undef
}

; Function Attrs: nounwind
declare i8* @llvm.hpvm.createNode1D(i8*, i64) #3

; Function Attrs: nounwind uwtable
define dso_local %struct.out.Func2 @Func2_cloned(i32* in %In, i64 %Insize, i32* out %Out, i64 %Outsize) #2 {
; CHECK-NOT: @Func2_cloned
entry:
  %Func3_cloned.node = call i8* @llvm.hpvm.createNode1D(i8* bitcast (%struct.out.Func3 (i32*, i64, i32*, i64)* @Func3_cloned to i8*), i64 3)
  call void @llvm.hpvm.bind.input(i8* %Func3_cloned.node, i32 0, i32 0, i1 false)
  call void @llvm.hpvm.bind.input(i8* %Func3_cloned.node, i32 1, i32 1, i1 false)
  call void @llvm.hpvm.bind.input(i8* %Func3_cloned.node, i32 2, i32 2, i1 false)
  call void @llvm.hpvm.bind.input(i8* %Func3_cloned.node, i32 3, i32 3, i1 false)
  call void @llvm.hpvm.bind.output(i8* %Func3_cloned.node, i32 0, i32 0, i1 false)
  ret %struct.out.Func2 undef
}

; Function Attrs: nounwind
declare i8* @llvm.hpvm.createNode(i8*) #3

; Function Attrs: nounwind uwtable
define dso_local %struct.out.PipeRoot @PipeRoot_cloned(i32* in %In1, i64 %Insize1, i32* in %In2, i64 %InSize2, i32* out %Out, i64 %Outsize) #2 {
; CHECK-NOT: @PipeRoot_cloned
entry:
  %Func2_cloned.node = call i8* @llvm.hpvm.createNode(i8* bitcast (%struct.out.Func2 (i32*, i64, i32*, i64)* @Func2_cloned to i8*))
  call void @llvm.hpvm.bind.input(i8* %Func2_cloned.node, i32 0, i32 0, i1 false)
  call void @llvm.hpvm.bind.input(i8* %Func2_cloned.node, i32 1, i32 1, i1 false)
  call void @llvm.hpvm.bind.input(i8* %Func2_cloned.node, i32 2, i32 2, i1 false)
  call void @llvm.hpvm.bind.input(i8* %Func2_cloned.node, i32 3, i32 3, i1 false)
  call void @llvm.hpvm.bind.output(i8* %Func2_cloned.node, i32 0, i32 0, i1 false)
  ret %struct.out.PipeRoot undef
}

; Function Attrs: nounwind
declare void @llvm.hpvm.init() #3

; Function Attrs: nounwind
declare i8* @llvm.hpvm.launch(i8*, i8*, i1) #3

; Function Attrs: nounwind
declare void @llvm.hpvm.wait(i8*) #3

; Function Attrs: nounwind
declare void @llvm.hpvm.cleanup() #3

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="cpu-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind }
attributes #2 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="cpu-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}
!hpvm_hint_cpu = !{!2, !3, !4, !5}
!hpvm_hint_gpu = !{}
!hpvm_hint_spir = !{}
!hpvm_hint_cudnn = !{}
!hpvm_hint_promise = !{}
!hpvm_hint_cpu_gpu = !{}
!hpvm_hint_cpu_spir = !{}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 9.0.0 (https://gitlab.engr.illinois.edu/llvm/hpvm.git 6690f9e7e8b46b96aea222d3e85315cd63545953)"}
!2 = !{%struct.out.Func1 (i32*, i64, i32*, i64)* @Func1_cloned}
!3 = !{%struct.out.Func3 (i32*, i64, i32*, i64)* @Func3_cloned}
!4 = !{%struct.out.Func2 (i32*, i64, i32*, i64)* @Func2_cloned}
!5 = !{%struct.out.PipeRoot (i32*, i64, i32*, i64, i32*, i64)* @PipeRoot_cloned}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !12, i64 0}
!11 = !{!"Root", !12, i64 0, !13, i64 8, !12, i64 16, !13, i64 24, !12, i64 32, !13, i64 40}
!12 = !{!"any pointer", !8, i64 0}
!13 = !{!"long", !8, i64 0}
!14 = !{!11, !13, i64 8}
!15 = !{!11, !12, i64 16}
!16 = !{!11, !13, i64 24}
!17 = !{!11, !12, i64 32}
!18 = !{!11, !13, i64 40}
