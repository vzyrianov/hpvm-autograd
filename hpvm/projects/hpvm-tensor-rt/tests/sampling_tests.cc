
#include "tests.h"


int main() {

  llvm_hpvm_initTensorRt(0);

  UnitTestResults unitTestResults;
  
  testSampling(unitTestResults); 
  
  unitTestResults.printSummary();

  return 0;
}
