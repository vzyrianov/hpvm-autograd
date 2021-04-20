
#include "tests.h"


int main() {

  llvm_hpvm_initTensorRt(0);

  UnitTestResults unitTestResults;
  
  testPerforation(unitTestResults);
  
  unitTestResults.printSummary();

  return 0;
}
