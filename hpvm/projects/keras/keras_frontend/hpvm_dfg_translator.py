
import os
import sys
from keras_frontend.utils import *
from keras_frontend.hpvm_intrinsics import *


class HPVMTranslator:

  def __init__(self, dfg, weight_str, input_str, filter_names):
    self.dfg = dfg
    self.output_map = {}
    self.counter = 0
    self.weight_str = weight_str
    self.input_str = input_str
    self.filter_names = filter_names
    self.node_str = ""
    self.root_str = ""
    self.root_struct_str = ""
    self.main_func_str = ""
    self.tuner_main_func_str = ""
    self.file_header_str = ""
    self.hpvm_node_names = {}
    

   
    
  def getVariableName(self, cur_node):    
    output_var_name = "var_" + str(self.counter)
    self.counter += 1
    self.output_map[cur_node.layer_name] = output_var_name
    self.hpvm_node_names[output_var_name] = 1

    return output_var_name


  
  def isSkipLayer(self, layer_type):
    skip_layers = {}
    skip_layers["Flatten"] = 0
    skip_layers["Dropout"] = 0
    skip_layers["ZeroPadding2D"] = 0

    if layer_type in skip_layers:
      return True
    else:
      return False

    
  
  # NOTE: returns the previous DFG node ignoring "Flatten", "Dropout" Layers
  def getPrevActiveLayer(self, cur_node):
    pred_layer_type = cur_node.inputs[0].layer_type
    # FIXME: Assuming the 'inference' phase - hence skipping Dropout
    #if pred_layer_type == "Flatten" or pred_layer_type == "Dropout":
    if self.isSkipLayer(pred_layer_type):
      cur_node = self.getPrevActiveLayer(cur_node.inputs[0])
      return cur_node
    else:
      return cur_node
    
      
  def getSingleInputName(self, cur_node):
    DEBUG (cur_node.layer_name)
    # Assumption: If no inputs, the previous layer must be input layer
    if len(cur_node.inputs) == 0:
      return "input"

    DEBUG ("Input_type = ", cur_node.inputs[0].layer_type)

    pred_layer_type = cur_node.inputs[0].layer_type
    # NOTE: Assuming the 'inference' phase - hence skipping Dropout
    #if pred_layer_type == "Flatten" or pred_layer_type == "Dropout":
    if self.isSkipLayer(pred_layer_type):
      cur_node = self.getPrevActiveLayer(cur_node)

    if cur_node.inputs[0].layer_type == "InputLayer":
      return "input"
  
    # get input to the layer
    input_node_name = cur_node.inputs[0].layer_name  # get the input layer ID

    input_var_name = ""
    if input_node_name in self.output_map:
      input_var_name = self.output_map[input_node_name]
    else:
      print ("Input Var not found - Aborting....")
      sys.exit(0)
      
    return input_var_name



  def getPrevLayerPadding(self, cur_node):
    DEBUG (cur_node.layer_name)
    # Assumption: If no inputs, the previous layer must be input layer
    if len(cur_node.inputs) == 0:
      return None

    DEBUG ("Input_type = ", cur_node.inputs[0].layer_type)
    if cur_node.inputs[0].layer_type == "ZeroPadding2D": 
      pred_padding = cur_node.inputs[0].padding
      return pred_padding
      
    return None

  

  def getMultipleInputNames(self, cur_node):
    var_names = []    
    for i in range(len(cur_node.inputs)):
      # get input to the layer
      input_node_name = cur_node.inputs[i].layer_name  # get the input layer ID

      input_var_name = ""
      if input_node_name in self.output_map:
        input_var_name = self.output_map[input_node_name]
        var_names.append(input_var_name)
      else:
        print ("Input Var not found - Aborting....")
        sys.exit(0)
      
    return var_names
  

  
  def hasBiasAdd(self, cur_node):
    if cur_node.layer_type == "Conv2D" or cur_node.layer_type == "Dense":
      return cur_node.use_bias

    return False


  def hasActivation(self, cur_node):
    if cur_node.layer_type == "Conv2D" or cur_node.layer_type == "Dense":
      return  cur_node.activation_type != "linear" 

    return False


  def genActivationCall(self, input_var, output_var, activation_type):
    header_str = self.genNodeHeader(output_var, 1)
    inst_str = header_str 

    func_name = ""
    if activation_type == "tanh":
      func_name += HPVM_tensor_tanh

    if activation_type == "relu":
      func_name += HPVM_tensor_relu

    if activation_type == "softmax":
      func_name += HPVM_tensor_softmax

    inst_str += "  void* r = " + func_name + "(t1); \n"
    footer_str = self.genNodeFooter(2)
    inst_str += footer_str
    
    return inst_str


  def genNodeHeader(self, var_name, num_params):
    node_header_str = "void " + var_name + "_node("
    for i in range(num_params):
      node_header_str += "void* t" + str(i + 1) + ", "
      node_header_str += "size_t bytes_t" + str(i + 1)
      if i < num_params - 1:
        node_header_str += ", "
        
    node_header_str += ") { \n" 

    node_header_str += " " + HPVM_hint + "(" + HPVM_layer_hint  + "); \n"
    node_header_str += " " + HPVM_attributes + "(" + str(num_params) + ", "
    
    for i in range(num_params):
      node_header_str += "t" + str(i + 1) 
      if i < num_params - 1:
        node_header_str += ", "
          
    node_header_str += ", 0); \n"

    # Adding node.id calls to assign IDs that are used with the runtime (for correct config ordering)
    node_header_str += " " + HPVM_node_id + "(" + str(self.counter) + "); \n\n"
    
    return node_header_str

    
  def genNodeFooter(self, num_params):

    node_footer_str = " " + HPVM_return + "("
    node_footer_str += str(num_params) + ", "
    node_footer_str += "r, "
    node_footer_str += "(size_t) 0); \n"
    node_footer_str += "}\n\n"

    return node_footer_str


  def genHpvmNodeEdges2(self, hpvm_node_id, input_vars):

    hpvm_edge_str = "\n  void* " + hpvm_node_id + " = "
    hpvm_edge_str += HPVM_createNodeND + "(0, " + hpvm_node_id + "_node); \n\n"
    
    it = 0
    for input_var_name in input_vars:
      if input_var_name in self.filter_names:
        input_index = self.filter_names[input_var_name]
        index1 = input_index * 2
        index2 = index1 + 1      
        hpvm_edge_str += " " + HPVM_bindIn + "(" + hpvm_node_id + ", " + str(index1) + ", " + str(it*2) + ", 0); \n"
        hpvm_edge_str += " " + HPVM_bindIn + "(" + hpvm_node_id + ", " + str(index2) + ", " + str(it*2+1) + ", 0); \n"

      elif input_var_name in self.hpvm_node_names:
        hpvm_edge_str += "  " + HPVM_edge + "(" + input_var_name + ", " + hpvm_node_id + ", 1, 0, " + str(it*2) + ", 0); \n"
        hpvm_edge_str += "  " + HPVM_edge + "(" + input_var_name + ", " + hpvm_node_id + ", 1, 1, " + str(it*2+1) + ", 0); \n"        
        
      it += 1
      
    return hpvm_edge_str


  
  def genHpvmNodeEdges(self, out_var_name, input_var_name, input_var_name2):

    DEBUG ("input_var_name2 = ", input_var_name2)
    DEBUG ("input_var_name = ", input_var_name)
    
    hpvm_edge_str = "\n  void* " + out_var_name + " = "
    hpvm_edge_str += HPVM_createNodeND + "(0, " + out_var_name + "_node); \n\n"

    if input_var_name in self.filter_names:
      input_index = self.filter_names[input_var_name]
      index1 = input_index * 2
      index2 = index1 + 1      
      hpvm_edge_str += " " + HPVM_bindIn + "(" + out_var_name + ", " + str(index1) + ", 0, 0); \n"
      hpvm_edge_str += " " + HPVM_bindIn + "(" + out_var_name + ", " + str(index2) + ", 1, 0); \n"

    elif input_var_name in self.hpvm_node_names:
      hpvm_edge_str += " " + HPVM_edge + "(" + input_var_name + ", " + out_var_name + ", 1, 0, 0, 0); \n"
      hpvm_edge_str += " " + HPVM_edge + "(" + input_var_name + ", " + out_var_name + ", 1, 1, 1, 0); \n"


    if input_var_name2 in self.filter_names:
      input_index = self.filter_names[input_var_name2]
      index1 = input_index * 2
      index2 = index1 + 1
      hpvm_edge_str += " " + HPVM_bindIn + "(" + out_var_name + ", " + str(index1) + ", 2, 0); \n"
      hpvm_edge_str += " " + HPVM_bindIn + "(" + out_var_name + ", " + str(index2) + ", 3, 0); \n"

    elif input_var_name2 in self.hpvm_node_names:
      hpvm_edge_str += " " + HPVM_edge + "(" + input_var_name2 + ", " + out_var_name + ", 1, 0, 2, 0); \n"
      hpvm_edge_str += " " + HPVM_edge + "(" + input_var_name2 + ", " + out_var_name + ", 1, 1, 3, 0); \n"

      
    return hpvm_edge_str

  

  def genDenseNode(self, cur_node):
    out_var_name = self.getVariableName(cur_node)    

    header_str = self.genNodeHeader(out_var_name, 2)
    inst_str = header_str 
    inst_str += "  void *r = " + HPVM_tensor_mul + "(t1, t2); \n"
    footer_str = self.genNodeFooter(2)
    inst_str += footer_str
    
    input_var_name = self.getSingleInputName(cur_node)
    weight_name = cur_node.layer_name + "_w"
    
    self.root_str +=  self.genHpvmNodeEdges(out_var_name, input_var_name, weight_name)
    
    self.node_str += inst_str

    
  

  def genConvNode(self, cur_node):

    out_var_name = self.getVariableName(cur_node)
    
    header_str = self.genNodeHeader(out_var_name, 2)
    inst_str = header_str 

    weights = cur_node.weights
    strides = cur_node.strides

    padding = 0
    if cur_node.padding.strip() == "valid":
      padding = 0
    else:
      padding = cur_node.padding      
      padding = int((weights.shape[0] - 1) / 2)

    prev_padding = self.getPrevLayerPadding(cur_node)
    if prev_padding != None:
      # FIXME: currently only supporting symmetric padding
      padding = prev_padding[0][0]        
      
    inst_str += "  void *r = " + HPVM_tensor_convolution + "(t1, t2, "
    inst_str += str(padding) + ", "
    inst_str += str(padding) + ", "
    inst_str += str(strides[0]) + ", "
    inst_str += str(strides[1]) 
    inst_str += "); \n"

    footer_str = self.genNodeFooter(2)
    inst_str += footer_str
        
    self.node_str += inst_str

    input_var_name = self.getSingleInputName(cur_node)
    weight_name = cur_node.layer_name + "_w"
    
    self.root_str +=  self.genHpvmNodeEdges(out_var_name, input_var_name, weight_name)


  def genDepthwiseConvNode(self, cur_node):

    out_var_name = self.getVariableName(cur_node)
    
    header_str = self.genNodeHeader(out_var_name, 2)
    inst_str = header_str 

    weights = cur_node.weights
    strides = cur_node.strides

    padding = 0
    if cur_node.padding.strip() == "valid":
      padding = 0
    else:
      padding = cur_node.padding      
      padding = int((weights.shape[0] - 1) / 2)

    prev_padding = self.getPrevLayerPadding(cur_node)
    if prev_padding != None:
      # FIXME: currently only supporting symmetric padding
      padding = prev_padding[0][0]        
      
    inst_str += "  void *r = " + HPVM_tensor_group_convolution + "(t1, t2, "
    inst_str += str(padding) + ", "
    inst_str += str(padding) + ", "
    inst_str += str(strides[0]) + ", "
    inst_str += str(strides[1]) + ", " 
    inst_str += "1, " 

    C = weights.shape[2]
    inst_str += str(C) + "); \n"

    footer_str = self.genNodeFooter(2)
    inst_str += footer_str
        
    self.node_str += inst_str

    input_var_name = self.getSingleInputName(cur_node)
    weight_name = cur_node.layer_name + "_w"
    
    self.root_str +=  self.genHpvmNodeEdges(out_var_name, input_var_name, weight_name)

    


  def genBatchNormNode(self, cur_node):

    out_var_name = self.getVariableName(cur_node)
    
    header_str = self.genNodeHeader(out_var_name, 5)
    inst_str = header_str 

    inst_str += "  void *r = " + HPVM_tensor_batchnorm + "(t1, t2, t3, t4, t5, "
    inst_str += str(cur_node.epsilon) + "); \n"

    footer_str = self.genNodeFooter(2)
    inst_str += footer_str
        
    self.node_str += inst_str

    layer_name = cur_node.layer_name
    input_id = self.getSingleInputName(cur_node)
    gamma_id = layer_name + "_gamma"
    beta_id = layer_name + "_beta"
    mean_id = layer_name + "_mean"
    variance_id = layer_name + "_variance"
    
    input_vars = []
    input_vars.append(input_id)
    input_vars.append(gamma_id)
    input_vars.append(beta_id)
    input_vars.append(mean_id)
    input_vars.append(variance_id)

    self.root_str +=  self.genHpvmNodeEdges2(out_var_name, input_vars)


    

  def genBiasNode(self, cur_node):
    input_var_name = self.output_map[cur_node.layer_name]
    out_var_name = self.getVariableName(cur_node)    

    header_str = self.genNodeHeader(out_var_name, 2)
    inst_str = header_str 
    inst_str += "  void *r = " + HPVM_tensor_add + "(t1, t2); \n"
    footer_str = self.genNodeFooter(2)
    inst_str += footer_str
        
    self.node_str += inst_str
    
    weight_name = cur_node.layer_name + "_b"    
    self.root_str +=  self.genHpvmNodeEdges(out_var_name, input_var_name, weight_name)

    
  
  def genSubActivationNode(self, cur_node):
    input_var_name = self.output_map[cur_node.layer_name] 
    out_var_name = self.getVariableName(cur_node)    
    activation_type = cur_node.activation_type
    
    inst_str = self.genActivationCall(input_var_name, out_var_name, activation_type)

    self.node_str += inst_str  

    self.root_str +=  self.genHpvmNodeEdges(out_var_name, input_var_name, "")
    


  def genActivationNode(self, cur_node):
    input_var_name = self.getSingleInputName(cur_node)
    out_var_name = self.getVariableName(cur_node)    
    activation_type = cur_node.activation_type
    
    inst_str = self.genActivationCall(input_var_name, out_var_name, activation_type)
    self.node_str += inst_str  

    self.root_str +=  self.genHpvmNodeEdges(out_var_name, input_var_name, "")


  def genAddNode(self, cur_node):
    out_var_name = self.getVariableName(cur_node)    
      
    header_str = self.genNodeHeader(out_var_name, 2)
    inst_str = header_str 
    inst_str += "  void *r = " + HPVM_tensor_add + "(t1, t2); \n"
    footer_str = self.genNodeFooter(2)
    inst_str += footer_str

    self.node_str += inst_str

    input_vars = self.getMultipleInputNames(cur_node)
    self.root_str +=  self.genHpvmNodeEdges(out_var_name, input_vars[0], input_vars[1])

    
    
  def genPoolNode(self, cur_node):
    out_var_name = self.getVariableName(cur_node)    

    header_str = self.genNodeHeader(out_var_name, 1)
    inst_str = header_str 

    pool_size = cur_node.pool_size
    strides = cur_node.strides
    # FIXME: Non-same padding is *NOT* currently supported
    padding = 0
    pool_type = 0
    func_name = ""

    layer_type = cur_node.layer_type
    if layer_type == "MaxPooling2D":
      func_name = HPVM_tensor_pool_max
    if layer_type == "AveragePooling2D":
      func_name = HPVM_tensor_pool_mean
      
    inst_str += "  void* r = " + func_name + "(t1, "
    inst_str += str(pool_size[0]) + ", " + str(pool_size[1]) + ", "
    inst_str += str(padding) + ", " + str(padding) + ", "
    inst_str += str(strides[0]) + ", " + str(strides[1]) + "); \n"
 
    footer_str = self.genNodeFooter(2)
    inst_str += footer_str
    
    self.node_str += inst_str

    input_var_name = self.getSingleInputName(cur_node)
    self.root_str +=  self.genHpvmNodeEdges(out_var_name, input_var_name, "")

    
  # Checks for call-type and redirects to functions that generate HPVM nodes
  def genHpvmNodes(self, cur_node):

    layer_type = cur_node.layer_type
        
    if layer_type == "Conv2D":
      self.genConvNode(cur_node)      

    if layer_type == "DepthwiseConv2D":
      self.genDepthwiseConvNode(cur_node)

    if layer_type == "BatchNormalization":
      self.genBatchNormNode(cur_node)

    if layer_type == "Dense":
      self.genDenseNode(cur_node)      

    if nodeHasBias(cur_node):  
      self.genBiasNode(cur_node)

    if nodeHasActivation(cur_node) and layer_type != "Activation":  
      self.genSubActivationNode(cur_node)     
      
    if layer_type == "Activation":
      self.genActivationNode(cur_node)     
      
    if layer_type == "Add":
      self.genAddNode(cur_node)
      
    if layer_type == "MaxPooling2D" or layer_type == "AveragePooling2D":  
      self.genPoolNode(cur_node)

      
          
     
  def codegenNode(self, dfg, cur_node, visited_nodes):

    # Skip visited nodes
    if cur_node.layer_name in visited_nodes:
      return
      
    if dfg.predVisited(cur_node, visited_nodes):      
      visited_nodes[cur_node.layer_name] = True

      self.genHpvmNodes(cur_node)
      # Invoking traversal on outbound nodes
      for output_node in cur_node.outputs:
        self.codegenNode(dfg, output_node, visited_nodes)
      

        
  # Print the DFG in reverse postorder
  def codegen(self, dfg):
    print ("\n *** Starting Codegen for ApproxHPVM DFG Representation *** \n")
    visited_nodes = {}
    # Starting traversal at the DFG root node
    self.codegenNode(dfg, dfg.root_node, visited_nodes)

    print ("\n --- Codegen Completed --- \n")


      

  def genFileHeader(self):
    headers = "\n#include <stdio.h> \n"
    headers += "#include <stdlib.h> \n"
    headers += "#include <unistd.h> \n"
    headers += "#include <fcntl.h> \n"
    headers += "#include <sys/stat.h> \n"
    headers += "#include <cstring> \n"
    
    headers += "#include <" + HPVM_header +  "> \n"
    headers += "#include <tensorUtils.h> \n\n"

    self.file_header_str = headers
        


  def genRootNodeHeader(self):
    root_signature = "void root("
    index = 0
    for f_name in self.filter_names:
      if index > 0:
        root_signature += "\t  "
      self.filter_names[f_name] = index
      root_signature += "void* " + f_name + ", "
      root_signature += "size_t " + f_name + "_bytes" 
      if index < len(self.filter_names) - 1:
        root_signature += ", \n"
      index += 1

    root_signature += "){ \n\n"

    root_signature += "\n  " + HPVM_hint +  "(" + HPVM_cpu_hint + "); \n"
    root_signature += " " + HPVM_attributes + "(" + str(len(self.filter_names)) + ", "
    
    index = 0
    for f_name in self.filter_names:
      root_signature += f_name 
      if index < len(self.filter_names) - 1:
        root_signature += ", "
      index += 1

    root_signature += ", 0); \n\n"
    
    self.root_str += root_signature


  def genRootNodeFooter(self):    
    last_node = self.dfg.last_node
    output_var = self.output_map[last_node.layer_name]

    # Binding output of last DFG node to the Root Node output
    root_footer_str = "\n  " + HPVM_bindOut + "(" + output_var + ", 0, 0, 0); \n"
    root_footer_str += "  " + HPVM_bindOut + "(" + output_var + ", 1, 1, 0); \n"
    root_footer_str += "\n}\n\n"
    
    self.root_str += root_footer_str
    

    
  def genRootStructure(self):
    root_struct = ""
    root_struct += "struct ret_t {\n"
    root_struct += "  void* tensor; \n"
    root_struct += "  size_t bytes; \n"
    root_struct += "}; \n\n"
    
    root_struct += "typedef struct __attribute__((__packed__)) {\n"
    for f_name in self.filter_names:
      root_struct += "  void* " + f_name + "; \n"
      root_struct += "  size_t " + f_name + "_bytes; \n"
      
    root_struct += "\n  struct ret_t r; \n"
    root_struct += "}\nRootIn;\n\n"

    self.root_struct_str += root_struct



  def genBatchLoop(self, data_shape, batch_size):

    chans = data_shape[1]
    width = data_shape[2]
    height = data_shape[3]    
    test_input_size = data_shape[0]

    func_str = "unsigned int batch_size = " + str(batch_size) + "; \n"
    func_str += "unsigned int test_input_size = " +  str(test_input_size) +  "; \n"
    func_str += "unsigned int batch_count = test_input_size / batch_size; \n\n"

    func_str += "startMemTracking(); \n"
    func_str += "startProfiling(); \n\n"
   
    func_str += "for(unsigned int j = 0; j < 1; j++){ \n"
    func_str += "#pragma clang loop unroll(disable) \n"
    func_str += "for(unsigned int i = 0; i < batch_count; i++){  \n\n"

    func_str += "unsigned int start = i * batch_size; \n"
    func_str += "unsigned int end = (i + 1) * batch_size;  \n"

    return func_str 
    

  def genBatchInput(self, data_shape, input_pth):

    chans = data_shape[1]
    width = data_shape[2]
    height = data_shape[3]    

    func_str = "void* input = readInputBatch(" + input_pth + ".c_str(), 0, start, end," + str(chans) + "," + str(width) + "," + str(height) +  ");  \n\n"
   
    func_str += "args->input = input;  \n"
    func_str += "args->input_bytes = 0; \n\n"

    return func_str
 

   
  def endBatchLoop(self):

    func_str = "freeBatchMemory(); \n"
    func_str += "} \n"
    func_str += "} \n\n"
    func_str += "stopProfiling();  \n"

    return func_str

  
  def handleTuneTestData(self):

    input_str = "void* input = test_input; \n"
    input_str += "std::string input_path = test_input_path; \n"
    input_str += "std::string labels_path = test_labels_path; \n\n"

    input_str += "if (runtype ==  \"tune\"){ \n"
    input_str += "  input = tune_input; \n"
    input_str += "  input_path = tune_input_path; \n"
    input_str += "  labels_path = tune_labels_path; \n\n"
    input_str += "} \n\n" 
    
    return input_str

  
    
  def genMainFunction(self, test_data, batch_size):

     main_func_str = "int main(int argc, char* argv[]){ \n\n"

     main_func_str += self.GetOptLoop()
     
     main_func_str += self.weight_str
     main_func_str += self.input_str
     main_func_str += "\n" + HPVM_init + "(); \n"

     main_func_str += """

if(config_path != ""){
  llvm_hpvm_initializeRuntimeController(config_path.c_str());
} 

 """
     
     main_func_str += "RootIn* args = static_cast<RootIn*>(malloc(sizeof(RootIn))); \n\n"

     main_func_str += self.handleTuneTestData()  
 
     for f_name in self.filter_names:    
       main_func_str += "args->" + f_name + " = " + f_name + "; \n"
       main_func_str += "args->" + f_name + "_bytes = 0; \n"       
    
     main_func_str += self.genBatchLoop(test_data.shape, batch_size)
     main_func_str += self.genBatchInput(test_data.shape, "input_path")
    
     main_func_str += "void* dfg = " + HPVM_launch + "(0, root, (void*) args); \n\n"
     main_func_str += HPVM_wait + "(dfg); \n\n"

     if LLVM_4_BRANCH:
       main_func_str += "void *result = static_cast<RootIn*>(args)->input; \n"
     elif LLVM_9_BRANCH:
       main_func_str += "void *result = static_cast<RootIn *>(args)->r.tensor; \n"
    
     main_func_str += "hpvm_request_tensor(result, 0); \n\n"
     main_func_str += "llvm_hpvm_invokeRtControl(result, labels_path.c_str(), start, end); \n"
  
     main_func_str += self.endBatchLoop()
     main_func_str += HPVM_cleanup + "(); \n "
  
     main_func_str += "return 0; \n\n"
     main_func_str += "} \n"    
    
     self.main_func_str += main_func_str



  def genTunerMainFunction(self, src_dir, test_data, batch_size):    

     tuner_main_func_str = "int main(int argc, char* argv[]){ \n\n"

     tuner_main_func_str += self.GetOptLoop()
     
     tuner_main_func_str += self.weight_str
     tuner_main_func_str += self.input_str
     tuner_main_func_str += "RootIn* args = static_cast<RootIn*>(malloc(sizeof(RootIn))); \n\n"

     tuner_main_func_str += self.handleTuneTestData()  
 
     for f_name in self.filter_names:    
       tuner_main_func_str += "args->" + f_name + " = " + f_name + "; \n"
       tuner_main_func_str += "args->" + f_name + "_bytes = 0; \n"       

     tuner_main_func_str += "\nint ret = 0; \n"
     tuner_main_func_str += "while ((ret = fifo_wait())) { \n"
     tuner_main_func_str += "\n" + HPVM_init + "(); \n"

     tuner_main_func_str += """

if(config_path != ""){
  llvm_hpvm_initializeRuntimeController(config_path.c_str());
} 
 
"""
     
     tuner_main_func_str += "std::string input_pth = (ret == 1 ? test_input_path : tune_input_path); \n"
     tuner_main_func_str += "std::string labels_pth = (ret == 1 ? test_labels_path : tune_labels_path); \n"

     abs_src_path = str(os.getcwd()) + "/" + src_dir 
     tuner_main_func_str += "auto* fp = open_fifo(\"" + abs_src_path + "/hpvm_fifo_w\", \"wb\"); \n\n"
     tuner_main_func_str += "float total_accuracy = 0; \n"
     
     tuner_main_func_str += self.genBatchLoop(test_data.shape, batch_size)
     tuner_main_func_str += self.genBatchInput(test_data.shape, "input_pth")

     tuner_main_func_str += "void* dfg = " + HPVM_launch + "(0, root, (void*) args); \n\n"
     tuner_main_func_str += HPVM_wait + "(dfg); \n\n"

     if LLVM_4_BRANCH:
       tuner_main_func_str += "void *result = static_cast<RootIn*>(args)->input; \n"
     elif LLVM_9_BRANCH:
       tuner_main_func_str += "void *result = static_cast<RootIn *>(args)->r.tensor; \n"
    
     tuner_main_func_str += "hpvm_request_tensor(result, 0); \n\n"
     tuner_main_func_str += "uint32_t* labels = readLabelsBatch3(labels_pth.c_str(), start, end); \n"
     tuner_main_func_str += "total_accuracy += computeAccuracy3(labels, result) * batch_size  ; \n"

     tuner_main_func_str += "\nfifo_write_batch(fp, result); \n"     

     tuner_main_func_str += self.endBatchLoop()

     tuner_main_func_str += "write_accuracy(total_accuracy / test_input_size); \n"
     tuner_main_func_str += "fclose(fp); \n"
     tuner_main_func_str += HPVM_cleanup + "(); \n "

     tuner_main_func_str += "\n}\n\n"  # End of FIFO loop
  
     tuner_main_func_str += "return 0; \n\n"
     tuner_main_func_str += "} \n"    
    
     self.tuner_main_func_str += tuner_main_func_str
     

  def addFIFORoutines(self, src_dir):

    abs_src_dir = str(os.getcwd()) + "/" + src_dir 

    FIFO_str = """
   
 FILE *open_fifo(const char *path, const char *mode) { 
  auto* fd = fopen(path, mode);
  if (!fd) {
    std::cerr << \"Error opening FIFO file: \" << strerror(errno);
    abort(); 
  }

   return fd;
}


int fifo_wait() {
    auto* fp = open_fifo(\"""" + abs_src_dir + """/hpvm_fifo_r\", \"r\");
    const int maxn = 100;
    char linebuf[maxn];
    fgets(linebuf, maxn, fp);
    fclose(fp);
    std::string line(linebuf);
    if (line == \"test\")
      return 1;
    if (line == \"tune\")
      return 2;
    if (line == \"stop\")
      return 0;
    std::cerr << \"Invalid fifo file content \" << line;
    abort();
}

void fifo_write_batch(FILE *fp, void *output_ptr) {
    auto *output = (Tensor *) output_ptr;
    const auto &dim = output->dims;
    size_t num_dims = dim.num_dims;
    fwrite(&num_dims, sizeof(size_t), 1, fp);
    fwrite(dim.dim_sizes, sizeof(size_t), dim.num_dims, fp);
    fwrite(output->host_data, 1, output->size_in_bytes, fp);
}


void write_accuracy(float accuracy) {
  std::ofstream fout("final_accuracy");
  fout << std::fixed << accuracy;
}

"""
    
    return FIFO_str
 
    
  def getUsageStr(self):

    usage_str = """

void printUsage(){
  std::cerr << \"Usage: -d {test|tune} -c {config_file_path} \";
  abort();
}

"""
    return usage_str
  


  def GetOptLoop(self):

    getopt_str = """

  std::string runtype;
  std::string config_path = "";
  int flag;
  while ( (flag = getopt (argc, argv, "hd:c:")) != -1){
    switch (flag)
      {
      case 'd':
	runtype = std::string(optarg);
	if (runtype != "test" && runtype != "tune")
	  printUsage();
	break;
      case 'c':
	config_path = std::string(optarg);
	break;
      case 'h':
	printUsage();
	break;
      default:
	printUsage(); 
      }
  }

"""
    
    return getopt_str
  
  

  def generateTestProgram(self, dir_prefix):
    
    program_str = self.file_header_str + self.node_str + self.root_str
    program_str += self.root_struct_str + self.getUsageStr() +  self.main_func_str

    DEBUG (program_str)
    
    f = open(dir_prefix + "/approxhpvm_src.cc", "w+")
    f.write(program_str)
    f.close()



  def generateTunerProgram(self, dir_prefix, FIFO_str):
    
    program_str = self.file_header_str + FIFO_str + self.node_str + self.root_str 
    program_str += self.root_struct_str + self.getUsageStr() + self.tuner_main_func_str

    DEBUG (program_str)
    
    f = open(dir_prefix + "/approxhpvm_tuner_src.cc", "w+")
    f.write(program_str)
    f.close()

    
  
  def translate(self, model, src_dir, test_data, tuner_data, batch_size):

    self.genFileHeader()
    
    self.genRootNodeHeader()
    self.genRootStructure()
    
    self.codegen(self.dfg)
    self.genRootNodeFooter()
    
    self.genMainFunction(test_data, batch_size)
    self.genTunerMainFunction(src_dir, test_data, batch_size)

    # dump generated program string to source file
    self.generateTestProgram(src_dir)

    FIFO_str = self.addFIFORoutines(src_dir)
    self.generateTunerProgram(src_dir, FIFO_str)
  

