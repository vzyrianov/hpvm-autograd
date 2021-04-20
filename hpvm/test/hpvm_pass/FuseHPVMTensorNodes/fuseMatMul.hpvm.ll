; RUN: opt -load LLVMBuildDFG.so -load LLVMInPlaceDFGAnalysis.so -load LLVMFuseHPVMTensorNodes.so -S -inplace -hpvm-fuse < %s | FileCheck %s
; ModuleID = 'fuseMatMul.ll'
source_filename = "fuseMatMul.cpp"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.out._Z11matmul_nodePvmS_m = type <{ i8*, i64 }>
%struct.out._Z13bias_add_nodePvmS_m = type <{ i8*, i64 }>
%struct.out._Z9tanh_nodePvm = type <{ i8*, i64 }>
%struct.out._Z13pool_max_nodePvm = type <{ i8*, i64 }>
%struct.out._Z4rootPvmS_mS_m = type <{ i8*, i64 }>

; CHECK-LABEL: @_Z13pool_max_nodePvm_cloned(
; CHECK: call i8* @llvm.hpvm.tensor.pool.max(
; CHECK: ret


; CHECK-LABEL: @_Z4rootPvmS_mS_m_cloned(
; CHECK: call i8* @llvm.hpvm.createNode(
; CHECK-NEXT: call void @llvm.hpvm.bind.input(
; CHECK-NEXT: call void @llvm.hpvm.bind.input(
; CHECK-NEXT: call void @llvm.hpvm.bind.input(
; CHECK-NEXT: call void @llvm.hpvm.bind.input(
; CHECK-NEXT: call void @llvm.hpvm.bind.input(
; CHECK-NEXT: call void @llvm.hpvm.bind.input(
; CHECK: call i8* @llvm.hpvm.createNode(
; CHECK-NEXT: call i8* @llvm.hpvm.createEdge(
; CHECK-NEXT: call i8* @llvm.hpvm.createEdge(
; CHECK: call void @llvm.hpvm.bind.output(
; CHECK: call void @llvm.hpvm.bind.output(
; CHECK: ret


; CHECK-LABEL: @_Z11matmul_nodePvmS_m_cloned__Z13bias_add_nodePvmS_m_cloned__Z9tanh_nodePvm_cloned(
; CHECK: call i8* @llvm.hpvm.tensor.mul(
; CHECK: call i8* @llvm.hpvm.tensor.add(
; CHECK: call i8* @llvm.hpvm.tensor.tanh(
; CHECK-NOT: call i8* @llvm.hpvm.tensor.pool.max(
; CHECK: ret





; Function Attrs: norecurse nounwind uwtable
define dso_local i32 @main() local_unnamed_addr #0 {
entry:
  call void @llvm.hpvm.init()
  %call = tail call noalias i8* @malloc(i64 64) #3
  %graphID = call i8* @llvm.hpvm.launch(i8* bitcast (%struct.out._Z4rootPvmS_mS_m (i8*, i64, i8*, i64, i8*, i64)* @_Z4rootPvmS_mS_m_cloned to i8*), i8* %call, i1 false)
  call void @llvm.hpvm.wait(i8* %graphID)
  %input = bitcast i8* %call to i8**
  %0 = load i8*, i8** %input, align 1, !tbaa !7
  tail call void @hpvm_request_tensor(i8* %0, i32 1) #3
  call void @llvm.hpvm.cleanup()
  ret i32 0
}

; Function Attrs: nofree nounwind
declare dso_local noalias i8* @malloc(i64) local_unnamed_addr #1

declare dso_local void @hpvm_request_tensor(i8*, i32) local_unnamed_addr #2

; Function Attrs: nounwind
declare i8* @llvm.hpvm.tensor.mul(i8*, i8*) #3

; Function Attrs: nounwind uwtable
define dso_local %struct.out._Z11matmul_nodePvmS_m @_Z11matmul_nodePvmS_m_cloned(i8* in %t1, i64 %bytes_t1, i8* in %t2, i64 %bytes_t2) #4 {
entry:
  %call1 = call i8* @llvm.hpvm.tensor.mul(i8* %t1, i8* %t2)
  %returnStruct = insertvalue %struct.out._Z11matmul_nodePvmS_m undef, i8* %call1, 0
  %returnStruct2 = insertvalue %struct.out._Z11matmul_nodePvmS_m %returnStruct, i64 0, 1
  ret %struct.out._Z11matmul_nodePvmS_m %returnStruct2
}

; Function Attrs: nounwind
declare i8* @llvm.hpvm.tensor.add(i8*, i8*) #3

; Function Attrs: nounwind uwtable
define dso_local %struct.out._Z13bias_add_nodePvmS_m @_Z13bias_add_nodePvmS_m_cloned(i8* in %t1, i64 %bytes_t1, i8* in %t2, i64 %bytes_t2) #4 {
entry:
  %call1 = call i8* @llvm.hpvm.tensor.add(i8* %t1, i8* %t2)
  %returnStruct = insertvalue %struct.out._Z13bias_add_nodePvmS_m undef, i8* %call1, 0
  %returnStruct2 = insertvalue %struct.out._Z13bias_add_nodePvmS_m %returnStruct, i64 0, 1
  ret %struct.out._Z13bias_add_nodePvmS_m %returnStruct2
}

; Function Attrs: nounwind
declare i8* @llvm.hpvm.tensor.tanh(i8*) #3

; Function Attrs: nounwind uwtable
define dso_local %struct.out._Z9tanh_nodePvm @_Z9tanh_nodePvm_cloned(i8* in %t1, i64 %bytes_t1) #4 {
entry:
  %call1 = call i8* @llvm.hpvm.tensor.tanh(i8* %t1)
  %returnStruct = insertvalue %struct.out._Z9tanh_nodePvm undef, i8* %call1, 0
  %returnStruct2 = insertvalue %struct.out._Z9tanh_nodePvm %returnStruct, i64 0, 1
  ret %struct.out._Z9tanh_nodePvm %returnStruct2
}

; Function Attrs: nounwind
declare i8* @llvm.hpvm.tensor.pool.max(i8*, i32, i32, i32, i32, i32, i32) #3

; Function Attrs: nounwind uwtable
define dso_local %struct.out._Z13pool_max_nodePvm @_Z13pool_max_nodePvm_cloned(i8* in %t1, i64 %bytes_t1) #4 {
entry:
  %call1 = call i8* @llvm.hpvm.tensor.pool.max(i8* %t1, i32 2, i32 2, i32 0, i32 0, i32 2, i32 2)
  %returnStruct = insertvalue %struct.out._Z13pool_max_nodePvm undef, i8* %call1, 0
  %returnStruct2 = insertvalue %struct.out._Z13pool_max_nodePvm %returnStruct, i64 0, 1
  ret %struct.out._Z13pool_max_nodePvm %returnStruct2
}

; Function Attrs: nounwind
declare i8* @llvm.hpvm.createNode(i8*) #3

; Function Attrs: nounwind
declare void @llvm.hpvm.bind.input(i8*, i32, i32, i1) #3

; Function Attrs: nounwind
declare i8* @llvm.hpvm.createEdge(i8*, i8*, i1, i32, i32, i1) #3

; Function Attrs: nounwind
declare void @llvm.hpvm.bind.output(i8*, i32, i32, i1) #3

; Function Attrs: nounwind uwtable
define dso_local %struct.out._Z4rootPvmS_mS_m @_Z4rootPvmS_mS_m_cloned(i8* in %input, i64 %input_bytes, i8* in %matmul2d_1_w, i64 %matmul2d_1_w_bytes, i8* in %matmul2d_1_b, i64 %matmul2d_1_b_bytes) #4 {
entry:
  %_Z11matmul_nodePvmS_m_cloned.node = call i8* @llvm.hpvm.createNode(i8* bitcast (%struct.out._Z11matmul_nodePvmS_m (i8*, i64, i8*, i64)* @_Z11matmul_nodePvmS_m_cloned to i8*))
  call void @llvm.hpvm.bind.input(i8* %_Z11matmul_nodePvmS_m_cloned.node, i32 0, i32 0, i1 false)
  call void @llvm.hpvm.bind.input(i8* %_Z11matmul_nodePvmS_m_cloned.node, i32 1, i32 1, i1 false)
  call void @llvm.hpvm.bind.input(i8* %_Z11matmul_nodePvmS_m_cloned.node, i32 2, i32 2, i1 false)
  call void @llvm.hpvm.bind.input(i8* %_Z11matmul_nodePvmS_m_cloned.node, i32 3, i32 3, i1 false)
  %_Z13bias_add_nodePvmS_m_cloned.node = call i8* @llvm.hpvm.createNode(i8* bitcast (%struct.out._Z13bias_add_nodePvmS_m (i8*, i64, i8*, i64)* @_Z13bias_add_nodePvmS_m_cloned to i8*))
  %output = call i8* @llvm.hpvm.createEdge(i8* %_Z11matmul_nodePvmS_m_cloned.node, i8* %_Z13bias_add_nodePvmS_m_cloned.node, i1 true, i32 0, i32 0, i1 false)
  %output1 = call i8* @llvm.hpvm.createEdge(i8* %_Z11matmul_nodePvmS_m_cloned.node, i8* %_Z13bias_add_nodePvmS_m_cloned.node, i1 true, i32 1, i32 1, i1 false)
  call void @llvm.hpvm.bind.input(i8* %_Z13bias_add_nodePvmS_m_cloned.node, i32 4, i32 2, i1 false)
  call void @llvm.hpvm.bind.input(i8* %_Z13bias_add_nodePvmS_m_cloned.node, i32 5, i32 3, i1 false)
  %_Z9tanh_nodePvm_cloned.node = call i8* @llvm.hpvm.createNode(i8* bitcast (%struct.out._Z9tanh_nodePvm (i8*, i64)* @_Z9tanh_nodePvm_cloned to i8*))
  %output2 = call i8* @llvm.hpvm.createEdge(i8* %_Z13bias_add_nodePvmS_m_cloned.node, i8* %_Z9tanh_nodePvm_cloned.node, i1 true, i32 0, i32 0, i1 false)
  %output3 = call i8* @llvm.hpvm.createEdge(i8* %_Z13bias_add_nodePvmS_m_cloned.node, i8* %_Z9tanh_nodePvm_cloned.node, i1 true, i32 1, i32 1, i1 false)
  %_Z13pool_max_nodePvm_cloned.node = call i8* @llvm.hpvm.createNode(i8* bitcast (%struct.out._Z13pool_max_nodePvm (i8*, i64)* @_Z13pool_max_nodePvm_cloned to i8*))
  %output4 = call i8* @llvm.hpvm.createEdge(i8* %_Z9tanh_nodePvm_cloned.node, i8* %_Z13pool_max_nodePvm_cloned.node, i1 true, i32 0, i32 0, i1 false)
  %output5 = call i8* @llvm.hpvm.createEdge(i8* %_Z9tanh_nodePvm_cloned.node, i8* %_Z13pool_max_nodePvm_cloned.node, i1 true, i32 1, i32 1, i1 false)
  call void @llvm.hpvm.bind.output(i8* %_Z13pool_max_nodePvm_cloned.node, i32 0, i32 0, i1 false)
  call void @llvm.hpvm.bind.output(i8* %_Z13pool_max_nodePvm_cloned.node, i32 1, i32 1, i1 false)
  ret %struct.out._Z4rootPvmS_mS_m undef
}

; Function Attrs: nounwind
declare void @llvm.hpvm.init() #3

; Function Attrs: nounwind
declare i8* @llvm.hpvm.launch(i8*, i8*, i1) #3

; Function Attrs: nounwind
declare void @llvm.hpvm.wait(i8*) #3

; Function Attrs: nounwind
declare void @llvm.hpvm.cleanup() #3

attributes #0 = { norecurse nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nofree nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #2 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #3 = { nounwind }
attributes #4 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="false" "no-infs-fp-math"="true" "no-jump-tables"="false" "no-nans-fp-math"="true" "no-signed-zeros-fp-math"="true" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="true" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}
!hpvm_hint_promise = !{!2, !3, !4, !5, !6}
!hpvm_hint_gpu = !{}
!hpvm_hint_cpu = !{}
!hpvm_hint_cpu_gpu = !{}
!hpvm_hint_cudnn = !{}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 9.0.0 (https://gitlab.engr.illinois.edu/llvm/hpvm.git 5ccb2a532b5a0d82cee5c0d29a629a29dec2307c)"}
!2 = !{%struct.out._Z11matmul_nodePvmS_m (i8*, i64, i8*, i64)* @_Z11matmul_nodePvmS_m_cloned}
!3 = !{%struct.out._Z13bias_add_nodePvmS_m (i8*, i64, i8*, i64)* @_Z13bias_add_nodePvmS_m_cloned}
!4 = !{%struct.out._Z9tanh_nodePvm (i8*, i64)* @_Z9tanh_nodePvm_cloned}
!5 = !{%struct.out._Z13pool_max_nodePvm (i8*, i64)* @_Z13pool_max_nodePvm_cloned}
!6 = !{%struct.out._Z4rootPvmS_mS_m (i8*, i64, i8*, i64, i8*, i64)* @_Z4rootPvmS_mS_m_cloned}
!7 = !{!8, !9, i64 0}
!8 = !{!"_ZTS6RootIn", !9, i64 0, !12, i64 8, !9, i64 16, !12, i64 24, !9, i64 32, !12, i64 40, !13, i64 48}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
!12 = !{!"long", !10, i64 0}
!13 = !{!"_ZTS5ret_t", !9, i64 0, !12, i64 8}
