#include <string>
#include <hpvm.h>
#include <tensorUtils.h>
#include <config.h>

#ifndef MODEL_PARAMS_DIR
#error MODEL_PARAMS_DIR is not defined
#endif

#define STR_VALUE(X) #X
#define STRINGIFY(X) STR_VALUE(X)
#define MODEL_PARAMS_DIR_STR STRINGIFY(MODEL_PARAMS_DIR)

void var_0_node(void *t1, size_t bytes_t1, void *t2, size_t bytes_t2) {
   __hpvm__hint(hpvm::TENSOR_TARGET);
   __hpvm__attributes(2, t1, t2, 0);

   void *r = __hpvm__tensor_add(t1, t2);
   __hpvm__return(2, r, (size_t)0);
}

void var_1_node(void *t1, size_t bytes_t1) {
   __hpvm__hint(hpvm::TENSOR_TARGET);
   __hpvm__attributes(1, t1, 0);

   void *r = __hpvm__tensor_tanh(t1);
   __hpvm__return(2, r, (size_t)0);
}

void root(void *input1, size_t input1_bytes, void *input2, size_t input2_bytes) {
   __hpvm__hint(hpvm::CPU_TARGET);
   __hpvm__attributes(4, input1, input1_bytes, input2, input2_bytes, 0);

   void* var_0 = __hpvm__createNodeND(0, var_0_node);

   __hpvm__bindIn(var_0, 0, 0, 0);
   __hpvm__bindIn(var_0, 1, 1, 0);
   __hpvm__bindIn(var_0, 2, 2, 0);
   __hpvm__bindIn(var_0, 3, 3, 0);

   void *var_1 = __hpvm__createNodeND(0, var_1_node);

   __hpvm__edge(var_0, var_1, 1, 0, 0, 0);
   __hpvm__edge(var_0, var_1, 1, 1, 1, 0);
   
   __hpvm__bindOut(var_1, 0, 0, 0);
   __hpvm__bindOut(var_1, 1, 1, 0);
}

struct ret_t {
   void *tensor;
   size_t bytes;
};

typedef struct __attribute__((__packed__)) {
   void *input1;
   size_t input1_bytes;

   void *input2;
   size_t input2_bytes;

   struct ret_t r;
} RootIn;

int main(int argc, char *argv[]) {


   RootIn *args = static_cast<RootIn *>(malloc(sizeof(RootIn)));
   void *input1 = create4DTensor(0, nchw, 1, 1, 28, 28);
   args->input1 = input1;
   args->input1_bytes = 0;

   void *input2 = create4DTensor(0, nchw, 1, 1, 28, 28);
   args->input2 = input2;
   args->input2_bytes = 0;
   __hpvm__init();

   //startMemTracking();

   void *dfg = __hpvm__launch(0, root, (void *)args);
   __hpvm__wait(dfg);
   void *result = static_cast<RootIn *>(args)->r.tensor;
   hpvm_request_tensor(result, 0);

   __hpvm__cleanup();
   return 0;
}
