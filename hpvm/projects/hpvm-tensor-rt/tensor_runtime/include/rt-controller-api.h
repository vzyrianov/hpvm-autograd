extern "C" {
// Functions to be inserted with initializeTensorRT and clearTensorRT
void llvm_hpvm_initializeRuntimeController(const char *);
void llvm_hpvm_clearRuntimeController();
void llvm_hpvm_invokeRtControl(void *result, const char *str, int start,
                               int end);
}
