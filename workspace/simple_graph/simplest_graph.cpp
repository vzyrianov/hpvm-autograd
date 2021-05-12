#include <iostream>
#include <string>
#include <hpvm.h>
#include <tensorUtils.h>
#include <config.h>

void var_0_node(void *t1, size_t bytes_t1) {
   __hpvm__hint(hpvm::TENSOR_TARGET);
   __hpvm__attributes(1, t1, 0);

   void *r = __hpvm__tensor_tanh(t1);
   __hpvm__return(2, r, (size_t)0);
}

void root(void *input1, size_t input1_bytes) {
   __hpvm__hint(hpvm::CPU_TARGET);
   __hpvm__attributes(2, input1, input1_bytes, 0);

   void* var_0 = __hpvm__createNodeND(0, var_0_node);

   __hpvm__bindIn(var_0, 0, 0, 0);
   __hpvm__bindIn(var_0, 1, 1, 0);

   __hpvm__bindOut(var_0, 0, 0, 0);
   __hpvm__bindOut(var_0, 1, 1, 0);
}

struct ret_t {
   void *tensor;
   size_t bytes;
};

typedef struct __attribute__((__packed__)) {
   void *input1;
   size_t input1_bytes;
 
  struct ret_t r;
} RootIn;

int main(int argc, char *argv[]) {


   RootIn *args = static_cast<RootIn *>(malloc(sizeof(RootIn)));
   void *input1 = create4DTensor(0, nchw, 1, 1, 2, 2);
   args->input1 = input1;
   args->input1_bytes = 0;

   {
      Tensor* tensor1 = static_cast<Tensor*>(input1);
      float* data = (float*) tensor1->host_data;
      data[0] = 0.7f;
      data[1] = 0.7f;
      data[2] = 0.7f;
      data[3] = 0.7f;
   }
   

   void* derivative = __hpvm__grad(root, (void*) args, 0);

   __hpvm__init();

   startMemTracking();

   void *dfg = __hpvm__launch(0, root, (void *)args);
   __hpvm__wait(dfg);
   void *result = static_cast<RootIn *>(args)->r.tensor;
   hpvm_request_tensor(result, 0);
   
   {
      Tensor* tensorOutput = static_cast<Tensor*>(result);
      float* data = (float*) tensorOutput->host_data;
      std::cout << "num_elems " << tensorOutput->num_elems << std::endl;
      std::cout << "size_in_bytes" << tensorOutput->size_in_bytes << std::endl;
      std::cout << "Output of tensor " << data[0] << " " << data[1] << " " << data[2] << " " << data[3] << std::endl;
   }

   __hpvm__cleanup();
   return 0;
}

