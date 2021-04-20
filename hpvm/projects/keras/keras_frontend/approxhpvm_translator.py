
import sys
import numpy as np
from keras_frontend.promise_translator import PromiseRtTranslator
from keras_frontend.hpvm_dfg_translator import HPVMTranslator
from keras_frontend.weight_utils import dumpLabels, dumpData, dumpConvWeights, dumpFcWeights, dumpFcBias
from keras_frontend.utils import *
from keras_frontend.knobs import *
import keras
import os



class DFG:

  root_set = False;

  def __init__(self):
    self.node_map = {}
    self.root_node = None
    self.last_node = None


  def hasSingleInput(self, layer):

    layer_name = layer.__class__.__name__
    
    singleInLayers = {}
    singleInLayers["DepthwiseConv2D"] = True
    singleInLayers["Conv2D"] = True
    singleInLayers["Dense"] = True   
    singleInLayers["MaxPooling2D"] = True   
    singleInLayers["Activation"] = True
    singleInLayers["BatchNormalization"] = True
    singleInLayers["Flatten"] = True   
    
    if layer_name in singleInLayers:
        return True

    return False


  def hasMultipleInputs(self, layer):
    layer_name = layer.__class__.__name__
    
    multipleInLayers = {}
    multipleInLayers["Add"] = True
    
    if layer_name in multipleInLayers:
        return True

    return False      



  def add_dfg_edge(self, inbound_node_name, dfg_node):

    inbound_node_name = inbound_node_name.split(":")[0]
    inbound_node_name = inbound_node_name.split("/")[0]
    if inbound_node_name in self.node_map:
      inbound_node = self.node_map[inbound_node_name]
      DEBUG (inbound_node_name, " found!")
      inbound_node.add_output(dfg_node)
      dfg_node.add_input(inbound_node)
      
    else:
      DEBUG ("--inbound node NOT FOUND!")

      

  
  def add_to_graph(self, layer):
    dfg_node = DFGNode(layer)
    if not self.root_set:
      self.root_node = dfg_node
      self.root_set = True # DFG root node is now set

    if self.hasMultipleInputs(layer):  
      for j in range(len(layer.input)):
        DEBUG (type(layer.input[j]))
        DEBUG (layer.input[j].op.name)        
        self.add_dfg_edge(layer.input[j].op.name, dfg_node)

    else:
      DEBUG (layer.input.name)        
      self.add_dfg_edge(layer.input.name, dfg_node)

    # Adding DFG node to name mapping
    self.node_map[layer.name] = dfg_node


  # Check if all predecessor nodes have been visited thus far - reverse postorder traversal
  def predVisited(self, cur_node, visited_nodes):
    for input_node in cur_node.inputs:
      if input_node.layer_name not in visited_nodes:
        return False;

    # All predecessors are visited 
    return True
      
    
  def traverseNode(self, cur_node, visited_nodes):

    # Skip visited nodes
    if cur_node.layer_name in visited_nodes:
      return
      
    if self.predVisited(cur_node, visited_nodes):
      DEBUG (cur_node.layer_type)
      DEBUG (cur_node.layer_name)
      visited_nodes[cur_node.layer_name] = True

      # Invoking traversal on outbound nodes
      for output_node in cur_node.outputs:
        self.traverseNode(output_node, visited_nodes)

      # NOTE: Assuming that no outbound edges implies the last node in the graph
      if len(cur_node.outputs) == 0:
        self.last_node = cur_node

        
  #Build and  Print the DFG in reverse postorder
  def buildDFG(self):
    DEBUG ("\n\n ****** Traversing and Printing DFG ******* \n\n")
    visited_nodes = {}
    # Starting traversal at the DFG root node
    self.traverseNode(self.root_node, visited_nodes)
    
       
      

class DFGNode:

    def add_output(self, output_node):
      self.outputs.append(output_node)
    
    def add_input(self, input_node):
      self.inputs.append(input_node)

      
    def __init__(self, layer):

      self.inputs = []
      self.outputs = []

      layer_type = layer.__class__.__name__
      self.layer_type = layer_type # layer type e.g., conv2d, add, dense
      self.layer_name = layer.name  # unique layer identifier
      DEBUG (self.layer_name)

      if layer_type == "Conv2D" or layer_type == "DepthwiseConv2D" or  layer_type == "Dense":
        self.weights = layer.get_weights()[0]
        DEBUG ("\t", self.weights.shape)
        self.use_bias = layer.use_bias
        
        if layer.use_bias:
          self.use_bias = layer.use_bias
          self.bias_weights = layer.get_weights()[1]
          DEBUG ("\t", self.bias_weights.shape)
        
          
      if layer_type == "Conv2D" or layer_type == "DepthwiseConv2D":
        self.padding = layer.padding
        self.strides = layer.strides
        DEBUG ("\t", self.strides)
        DEBUG ("\tPadding = ", self.padding)

        
      if layer_type == "MaxPooling2D" or layer_type == "AveragePooling2D":
        self.pool_size = layer.pool_size
        self.strides = layer.strides
        DEBUG ("\t pool_size = ", self.pool_size)
        DEBUG ("\t strides = ", self.strides)

        
      if layerHasActivationAttr(self):
        self.activation_type = layer.activation.__name__
        DEBUG ("\t Activation = ", self.activation_type)
  

      if layer_type == "ZeroPadding2D":
        DEBUG ("***ZeroPaddding \n");
        self.padding = layer.padding
        DEBUG ("padding = ", self.padding);
        
      if layer_type == "BatchNormalization":
        self.epsilon = layer.epsilon
        self.beta = layer.beta
        self.gamma = layer.gamma
        self.moving_mean = layer.moving_mean
        self.moving_variance = layer.moving_variance
        

        
        
        

class TensorRtTranslator:

  def __init__(self, dfg):
    self.dfg = dfg
    self.output_map = {}
    self.counter = 0
    self.weight_str = ""
    self.program_str = ""
    self.input_str = ""
    self.filter_names = {}

    # Used for Json gen
    self.json_str = ""
    self.knobs_str = ""
    self.cur_height = 32    
    self.cur_width = 32     
    self.op_count = 0       
    
    


  def setInputHeightWidth(self, data):

    self.cur_height = data.shape[2]
    self.cur_width = data.shape[3]
    DEBUG ("cur_height = ", self.cur_height, "  cur_width = ", self.cur_width, ", \n")

    
  def addConvOverheads(self, weights, padding, strides):

    K_d = weights.shape[0] * weights.shape[1] * weights.shape[2] * weights.shape[3]

    H_d = self.cur_height / strides[0]
    W_d = self.cur_width / strides[1]

    flops = H_d * W_d * K_d
    DEBUG ("conv_flops =  ", flops)

    self.json_str += "\"convolution_" + str(self.op_count) + "\" : " + str(flops) + ", \n"
    self.knobs_str += "\"convolution_" + str(self.op_count) + "\" : ["  + conv_knobs + "], \n"
    self.op_count += 1
    
    self.cur_height = self.cur_height / strides[0]
    self.cur_width = self.cur_width / strides[1]

    DEBUG ("cur_height = ", self.cur_height, "  cur_width = ", self.cur_width, "\n")

    
  def addDenseOverheads(self, weights):

    flops = weights.shape[0] * weights.shape[1]
    DEBUG ("dense_flops =  ", flops)

    self.json_str += "\"linear_" + str(self.op_count) + "\" : " + str(flops) + ", \n"
    self.knobs_str += "\"linear_" + str(self.op_count) + "\" : ["  + baseline_knobs + "], \n"
    self.op_count += 1
        
    self.cur_height = 1
    self.cur_width = weights.shape[1] 
    
    DEBUG ("cur_height = ", self.cur_height, "  cur_width = ", self.cur_width, "\n")

    
  def adjustPoolDims(self, strides):

    self.cur_height = self.cur_height / strides[0]
    self.cur_width = self.cur_width / strides[1]
    
    DEBUG ("cur_height = ", self.cur_height, "  cur_width = ", self.cur_width, "\n")


  def addBaselineKnob(self, op_name):

    self.json_str += "\"" + op_name + "_" + str(self.op_count) + "\" : 0, \n"
    self.knobs_str += "\"" + op_name + "_" + str(self.op_count) + "\" : ["  + baseline_knobs + "], \n"
    self.op_count += 1

    
    
    
  def getWeightStr(self):
    return self.weight_str


  def getInputStr(self):
    return self.input_str


  def getFilterNames(self):
    return self.filter_names

    
  def getWeightVarName(self, weights):
    
    output_var_name = "weights_" + str(self.w_counter)
    self.w_counter += 1
    self.filter_names[weights] = output_var_name

    return output_var_name

    
  def getVariableName(self, cur_node):
    
    output_var_name = "var_" + str(self.counter)
    self.counter += 1
    self.output_map[cur_node.layer_name] = output_var_name

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

    # NOTE: Assuming the 'inference' phase - hence skipping Dropout
    pred_layer_type = cur_node.inputs[0].layer_type
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

    if nodeHasBias(cur_node):
      return cur_node.use_bias

    return False


  def hasActivation(self, cur_node):

    if nodeHasActivation(cur_node):
      return cur_node.activation_type != "linear" 

    return False



  
      
  def genNodeCalls(self, cur_node):

    out_var_name1 = self.getVariableName(cur_node)    
    layer_type = cur_node.layer_type
        
    if layer_type == "Conv2D" or layer_type == "DepthwiseConv2D":
      input_var_name = self.getSingleInputName(cur_node)
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
      
      inst_str = "void* " + out_var_name1 + " = "
      inst_str += "tensorConvolution(" + input_var_name + ", "
      inst_str += cur_node.layer_name + "_w, "
      inst_str += str(padding) + ", "
      inst_str += str(padding) + ", "
      inst_str += str(strides[0]) + ", "
      inst_str += str(strides[1]) + ", "
      inst_str += "1, "

      if layer_type == "DepthwiseConv2D":
        C = weights.shape[2]
        inst_str += str(C) + "); \n"
      else:
        inst_str += "1); \n"
        
      self.program_str += inst_str


      if strides[0] > 1 and cur_node.padding.strip() == "same":
        print ("!ERROR: Same Padding not supported for Conv with Stride > 1")
        print ("Use: ZeroPadding2D(padding=(" + str(padding) + "," + str(padding) + ")) before the Conv2D/DepthwiseConv2D Operator  \n");
        sys.exit(0)

      # NOTE: For Json (tuning config) file generation
      if layer_type == "Conv2D":
        self.addConvOverheads(weights, padding, strides)
  
      elif layer_type == "DepthwiseConv2D":
        self.addBaselineKnob("depthwise_convolution")

    

    if layer_type == "Dense":
      input_var_name = self.getSingleInputName(cur_node)

      weights = cur_node.weights
      inst_str = "void* " + out_var_name1 + " = "
      inst_str += "tensorGemmGPU(" + input_var_name + ", "
      inst_str += cur_node.layer_name + "_w"
      inst_str += "); \n"

      self.program_str += inst_str

      # Add Cost for Dense Layer (Json file)
      self.addDenseOverheads(weights)
        
      
    if self.hasBiasAdd(cur_node):
      out_var_name2 = self.getVariableName(cur_node)    

      inst_str = "void* " + out_var_name2 + " = "
      inst_str += "tensorAdd(" + out_var_name1 + ", "
      inst_str += cur_node.layer_name + "_b"
      inst_str += "); \n"

      self.program_str += inst_str

      # NOTE: Changing output variable
      out_var_name1 = out_var_name2

      #self.json_str += "add_" + str(self.op_count) + " : 0, \n"
      # self.op_count += 1
      self.addBaselineKnob("add")
      

    if layer_type == "Activation":
      input_var_name = self.getSingleInputName(cur_node)
      
      inst_str = genActivationCallStr(input_var_name, out_var_name1, cur_node.activation_type)
      self.program_str += inst_str

      #self.json_str += cur_node.activation_type + "_" + str(self.op_count) + " : 0, \n"
      #self.op_count += 1
      self.addBaselineKnob(cur_node.activation_type)

    
    if self.hasActivation(cur_node) and layer_type != "Activation":
      activation_type = cur_node.activation_type
      out_var_name3 = self.getVariableName(cur_node)    

      if activation_type == "softmax":
        print ("Softmax canNOT be part of Dense/Conv Op. Insert: Activation('softmax');")
        sys.exit(0)
        
      inst_str = genActivationCallStr(out_var_name1, out_var_name3, activation_type)
      self.program_str += inst_str  

      self.addBaselineKnob(activation_type)

        

    if layer_type == "BatchNormalization":
      input_var_name = self.getSingleInputName(cur_node)

      inst_str = "void* " + out_var_name1 + " = "
      inst_str += "tensorBatchNorm(" + input_var_name + ", "
      inst_str += cur_node.layer_name + "_gamma, "
      inst_str += cur_node.layer_name + "_beta, "
      inst_str += cur_node.layer_name + "_mean, "
      inst_str += cur_node.layer_name + "_variance, "
      inst_str += str(cur_node.epsilon)
      inst_str += "); \n"
      
      self.program_str += inst_str

      #self.json_str += "batchnorm_" + str(self.op_count) + " : 0, \n"
      #self.op_count += 1
      self.addBaselineKnob("batchnorm")

      
      
    if layer_type == "Add":  
      input_vars = self.getMultipleInputNames(cur_node)
      
      inst_str = "void* " + out_var_name1 + " = "
      inst_str += "tensorAdd(" + input_vars[0] + ", " + input_vars[1] + "); \n"
      self.program_str += inst_str

      #self.json_str += "add_" + str(self.op_count) + " : 0, \n"
      #self.op_count += 1
      self.addBaselineKnob("add")

      
    if layer_type == "MaxPooling2D" or layer_type == "AveragePooling2D":  
      input_var_name = self.getSingleInputName(cur_node)

      pool_size = cur_node.pool_size
      strides = cur_node.strides
      # FIXME: Non-same padding is *NOT* currently supported
      padding = 0
      pool_type = 0
      if layer_type == "MaxPooling2D":
        pool_type = "0"
        #self.json_str += "maxpool_" + str(self.op_count) + " : 0, \n"
        #self.op_count += 1
        self.addBaselineKnob("maxpool")

      if layer_type == "AveragePooling2D":
        pool_type = "1"
        #self.json_str += "avgpool_" + str(self.op_count) + " : 0, \n"
        #self.op_count += 1
        self.addBaselineKnob("avgpool")

      
      # tensorPooling(input, pool_type, pool_h, pool_w, v_pad, h_pad, v_stride, h_stride)
      inst_str = "void* " + out_var_name1 + " = "
      inst_str += "tensorPooling(" + input_var_name + "," + pool_type + "," + str(pool_size[0]) + "," + str(pool_size[1]) 
      inst_str +=  "," + str(padding) + "," + str(padding) + "," + str(strides[0]) + "," + str(strides[1])
      inst_str += "); \n"
      self.program_str += inst_str

      self.adjustPoolDims(strides)
      
            
          
     
  def codegenNode(self, dfg, cur_node, visited_nodes):

    # Skip visited nodes
    if cur_node.layer_name in visited_nodes:
      return

    DEBUG ("-visiting = ", cur_node.layer_name, "\n")
    
    if dfg.predVisited(cur_node, visited_nodes):
      
      visited_nodes[cur_node.layer_name] = True
      self.genNodeCalls(cur_node)

      # Invoking traversal on outbound nodes
      for output_node in cur_node.outputs:
        self.codegenNode(dfg, output_node, visited_nodes)
      
          
  # Print the DFG in reverse postorder
  def codegen(self, dfg):

    print ("\n *** Starting Codegen for HPVM Tensor Rt *** \n")
    visited_nodes = {}
    # Starting traversal at the DFG root node
    self.codegenNode(dfg, dfg.root_node, visited_nodes)

    print ("\n\n --- Codegen Completed --- \n\n")


    
    
  def dump_weights(self, model, prefix, reload_weights):

    layer_count = 0
    for i in range(len(model.layers)):
      layer = model.layers[i]
      layer_type = layer.__class__.__name__
      layer_name = layer.name

      if layer_type == "Conv2D" or layer_type == "DepthwiseConv2D":
        weights = layer.get_weights()[0]
        w_name = layer_name + "_w"
        
        self.filter_names[w_name] = 1
        DEBUG (weights.shape, w_name)

        N = weights.shape[3]
        C = weights.shape[2]
        H = weights.shape[1]
        W = weights.shape[0]

        unique_file_name = w_name + ".bin"
        dumpConvWeights(prefix + unique_file_name, weights, N, C, H, W, reload_weights)

        file_path = w_name + "_path" 
        file_path_str = "std::string " + file_path + " = " + " dir_prefix + std::string(\""
        file_path_str += unique_file_name + "\"); \n"
        self.weight_str += file_path_str

        # NOTE: Special handling for DepthwiseConv2D
        if layer_type == "DepthwiseConv2D":
          N = C
          C = 1   
        
        # FIXME: Be flexible for datatypes (currently only FP32 weights)
        # NOTE: '0' specified for floating point type
        self.weight_str += "void* " + w_name + " = " + " readTrainedWeights("
        self.weight_str += file_path + ".c_str(), 0," + str(N) + "," + str(C) + "," + str(H) + "," + str(W)
        self.weight_str += "); \n"
        
        
        if layer.use_bias:
          bias_weights = layer.get_weights()[1]
          b_name = layer_name + "_b"

          self.filter_names[b_name] = 1
          DEBUG (bias_weights.shape, b_name)

          unique_file_name = b_name + ".bin"
          dumpFcBias(prefix + unique_file_name, bias_weights, bias_weights.shape[0], reload_weights)

          file_path = b_name + "_path" 
          file_path_str =  "std::string " + file_path + " = " + " dir_prefix + std::string(\""
          file_path_str += unique_file_name + "\"); \n"
          self.weight_str += file_path_str

          C = bias_weights.shape[0]

          self.weight_str += "void* " + b_name + " = " + " readTrainedWeights("
          self.weight_str += file_path + ".c_str(), 0,1," + str(C) + ",1,1); \n"


      if layer_type == "Dense":
        weights = layer.get_weights()[0]
        w_name = layer_name + "_w"

        self.filter_names[w_name] = 1
        DEBUG (weights.shape, w_name)

        H = weights.shape[0]
        W = weights.shape[1]

        unique_file_name = w_name + ".bin"
        dumpFcWeights(prefix + unique_file_name, weights, H, W, reload_weights)

        file_path = w_name + "_path" 
        file_path_str = "std::string " + file_path + " = " + " dir_prefix + std::string(\""
        file_path_str += unique_file_name + "\"); \n"
        self.weight_str += file_path_str
     
        self.weight_str += "void* " + w_name + " = " + " readTrainedWeights("
        self.weight_str += file_path + ".c_str(), 0,1,1," + str(H) + "," + str(W) + "); \n"
        
        
        if layer.use_bias:
          bias_weights = layer.get_weights()[1]
          b_name = layer_name + "_b"

          self.filter_names[b_name] = 1
          DEBUG (bias_weights.shape, b_name)

          unique_file_name = b_name + ".bin"
          dumpFcBias(prefix + unique_file_name, bias_weights, bias_weights.shape[0], reload_weights)

          file_path = b_name + "_path" 
          file_path_str =  "std::string " + file_path + " = " + " dir_prefix + std::string(\"" 
          file_path_str += unique_file_name + "\"); \n"
          self.weight_str += file_path_str

          C = bias_weights.shape[0]

          self.weight_str += "void* " + b_name + " = " + " readTrainedWeights("
          self.weight_str += file_path + ".c_str(), 0,1," + str(C) + ",1,1); \n"
          

      if layer_type == "BatchNormalization":
        weights = layer.get_weights()
        
        gamma_w = weights[0]
        gamma_id = layer_name + "_gamma"
        gamma_file_name = gamma_id + ".bin"
        self.filter_names[gamma_id] = 1
        dumpFcBias(prefix + gamma_file_name, gamma_w, gamma_w.shape[0], reload_weights)

        file_path = gamma_id + "_path" 
        file_path_str =  "std::string " + file_path + " = " + " dir_prefix + std::string(\"" 
        file_path_str += gamma_file_name + "\"); \n"
        self.weight_str += file_path_str
        C = gamma_w.shape[0]
        self.weight_str += "void* " + gamma_id + " = " + " readTrainedWeights("
        self.weight_str += file_path + ".c_str(), 0,1," + str(C) + ",1,1); \n"
        # End of Gamma handling   
        
        beta_w = weights[1]
        beta_id = layer_name + "_beta"
        beta_file_name = beta_id + ".bin"
        self.filter_names[beta_id] = 1
        dumpFcBias(prefix + beta_file_name, beta_w, beta_w.shape[0], reload_weights)

        file_path = beta_id + "_path" 
        file_path_str =  "std::string " + file_path + " = " + " dir_prefix + std::string(\"" 
        file_path_str += beta_file_name + "\"); \n"
        self.weight_str += file_path_str
        self.weight_str += "void* " + beta_id + " = " + " readTrainedWeights("
        self.weight_str +=  file_path + ".c_str(), 0,1," + str(C) + ",1,1); \n"
        # End of Beta Handling       

        mean_w = weights[2]
        mean_id = layer_name + "_mean"
        mean_file_name = mean_id + ".bin"
        self.filter_names[mean_id] = 1
        dumpFcBias(prefix + mean_file_name, mean_w, mean_w.shape[0], reload_weights)
        
        file_path = mean_id + "_path" 
        file_path_str =  "std::string " + file_path + " = " + " dir_prefix + std::string(\"" 
        file_path_str += mean_file_name + "\"); \n"
        self.weight_str += file_path_str
        self.weight_str += "void* " + mean_id + " = " + " readTrainedWeights("
        self.weight_str += file_path + ".c_str(), 0,1," + str(C) + ",1,1); \n"
        # End of Mean Handling      
    
        
        variance_w = weights[3]
        variance_id = layer_name + "_variance"
        variance_file_name = variance_id + ".bin"
        self.filter_names[variance_id] = 1
        dumpFcBias(prefix + variance_file_name, variance_w, variance_w.shape[0], reload_weights)

        file_path = variance_id + "_path" 
        file_path_str =  "std::string " + file_path + " = " + " dir_prefix + std::string(\"" 
        file_path_str += variance_file_name + "\"); \n"
        self.weight_str += file_path_str
        self.weight_str += "void* " + variance_id + " = " + " readTrainedWeights("
        self.weight_str += file_path + ".c_str(), 0,1," + str(C) + ",1,1); \n"
        # End of Variance Handling      
            
      layer_count += 1


       

  def add_header(self):

    headers = "\n#include <stdio.h> \n"
    headers += "#include <stdlib.h> \n"
    headers += "#include <unistd.h> \n"
    headers += "#include <fcntl.h> \n"
    headers += "#include <sys/types.h> \n"
    headers += "#include <sys/stat.h> \n"
    headers += "#include <string.h> \n"

    headers += "#include \"tensor_runtime.h\" \n"
    headers += "#include \"utils.h\" \n\n"

    main_func = "int main(){ \n\n"

    initialization = "llvm_hpvm_initTensorRt(0); \n\n"
    
    self.program_str += headers
    self.program_str += main_func
    self.program_str += initialization
    

    
  def add_footer(self, test_data):

    if test_data is not None and self.dfg.last_node is not None:
      last_node = self.dfg.last_node
      output_var = self.output_map[last_node.layer_name]
    
      
    destructors = "\nllvm_hpvm_cleanupTensorRt(); \n"
    self.program_str += destructors
    
    end_main = "\nreturn 0; \n\n}\n"
    self.program_str += end_main
    
    return 0
    


  def genInputReadCall(self, input_data, input_name):

    file_path =  input_name + "_path" 
    file_path_str = "std::string " + file_path + " = " + " dir_prefix + std::string(\""
    file_path_str += input_name + ".bin\"); \n"
    self.weight_str += file_path_str
    
    N = input_data.shape[0]
    C = input_data.shape[1]
    H = input_data.shape[2]
    W = input_data.shape[3]

    self.input_str += "void* " + input_name +  " = readTrainedWeights("
    self.input_str += file_path + ".c_str(), 0," + str(N) + "," + str(C) + ","
    self.input_str += str(H) + "," + str(W) + "); \n"



  def genLabelReadCall(self, labels, labels_name):

    file_path = labels_name + "_path" 
    file_path_str = "std::string " + file_path + " = " + " dir_prefix + std::string(\""
    file_path_str +=  labels_name + ".bin\"); \n"
    self.weight_str += file_path_str

    self.input_str += "uint32_t* " + labels_name + " = readLabels3("
    self.input_str += file_path + ".c_str()," + str(labels.shape[0]) + "); \n"


    

  def genInputCalls(self, test_data, test_labels, tuner_data, tuner_labels, weights_dir, reload_weights):

    dumpData(weights_dir + "test_input.bin", test_data, reload_weights)
    self.genInputReadCall(test_data, "test_input")
    # Adding input to the filter map
    self.filter_names["input"] = 1
    dumpLabels(weights_dir + "test_labels.bin", test_labels, reload_weights)
    self.genLabelReadCall(test_labels, "test_labels")

    dumpData(weights_dir + "tune_input.bin", tuner_data, reload_weights)
    self.genInputReadCall(test_data, "tune_input")
 
    dumpLabels(weights_dir + "tune_labels.bin", tuner_labels, reload_weights)
    self.genLabelReadCall(test_labels, "tune_labels")



    

    
  def genBatchLoop(self, x_test):

    N = x_test.shape[0]
    C = x_test.shape[1]
    H = x_test.shape[2]
    W = x_test.shape[3]
    
    loop_str = ""
    loop_str += "\nstartMemTracking(); \n\n"
    
    loop_str += "int test_input_size = " + str(N) + "; \n"
    loop_str += "int batch_size = " + str(N) + "; \n"
    loop_str += "int batch_count = test_input_size / batch_size; \n"
    loop_str += "float final_accuracy = 0.0; \n\n"

    loop_str += "for(int i = 0; i < batch_count; i++){ \n\n"
    loop_str += "int start = i * batch_size; \n"
    loop_str += "int end = (i + 1) * batch_size; \n"

    loop_str += "\nvoid* input = readInputBatch(input_path.c_str(),0,start,end," 
    loop_str += str(C) + "," + str(H) + "," + str(W) + "); \n\n"

    self.program_str += loop_str


    
  def endBatchLoop(self):

    end_loop_str = ""
    
    end_loop_str += "\nuint32_t* labels = readLabelsBatch3(labels_path.c_str(),start,end); \n"

    last_node = self.dfg.last_node
    output_var = self.output_map[last_node.layer_name]
    accuracy_call = "\nfloat accuracy = computeAccuracy3(labels, " + output_var + "); \n"
    end_loop_str += accuracy_call
 
    end_loop_str += "final_accuracy += accuracy; \n"
    end_loop_str += "freeBatchMemory(); \n "
    end_loop_str += "\n}\n\n"

    end_loop_str += "final_accuracy = final_accuracy / batch_count; \n"
    end_loop_str += "dumpFinalAccuracy(final_accuracy); \n\n"

    self.program_str += end_loop_str
    
    
    

  def generateSourceProgram(self, dir_prefix):

    f = open(dir_prefix + "/src.cc", "w+")
    f.write(self.program_str)
    f.close()


  def dumpJsonFile(self, dir_prefix, weights_dir):

    f = open(dir_prefix + "/tuner.json", "w+")
    f.write("{ \n\n")
    
    op_cost_str = " \"op_cost\" : { \n"
    op_cost_str += self.json_str[:-3]
    op_cost_str += "\n }, \n\n"
    f.write(op_cost_str)
     
    f.write(knobs_str + " \n")    
    
    layer_knobs_str = " \"op_knobs\" : { \n"
    layer_knobs_str += self.knobs_str[:-3]
    layer_knobs_str += " \n\n }, \n\n"

    labels_path = weights_dir + "/tune_labels.bin"
    layer_knobs_str += "\"tune_labels_path\" : \"" + labels_path + "\", \n"
    layer_knobs_str += "\"conf_path\" : \"tuner_confs.txt\", \n"
    layer_knobs_str += "\"fifo_path_r\": \"hpvm_fifo_r\", \n"
    layer_knobs_str += "\"fifo_path_w\": \"hpvm_fifo_w\" \n"
    
    f.write(layer_knobs_str)

    f.write("\n\n}")
    f.close()

    
  
  def translate(self, model, weights_dir, src_dir, test_data, test_labels, tuner_data, tuner_labels, weights_reload):

    self.add_header()
    
    dir_path = "std::string dir_prefix = std::string(\"" + weights_dir +  "\"); \n"
    self.weight_str += dir_path

    if test_data is not None:
      self.genInputCalls(test_data, test_labels, tuner_data, tuner_labels, weights_dir, weights_reload)

    self.dump_weights(model, weights_dir, weights_reload)
    self.program_str += "\n" + self.weight_str + "\n\n"

    self.genBatchLoop(test_data)
    
    self.codegen(self.dfg)

    self.endBatchLoop()

    self.add_footer(test_data);

    self.generateSourceProgram(src_dir)
    
    self.dumpJsonFile(src_dir, weights_dir)
    



def reloadModelParams(model, reload_dir, x_test, y_test):

  print ("\n\n*****NOTE: Reloading pre-trained weights \n")

  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss2:', score[0])
  print('Test accuracy2:', score[1])

  for i in range(len(model.layers)):
    layer = model.layers[i]
    layer_name = layer.name
    DEBUG ("*layer_name = ", layer_name)

    if "conv" not in layer_name and "dense" not in layer_name:
      continue
    
    w_path = reload_dir + layer_name + "_w.bin"
    b_path = reload_dir + layer_name + "_b.bin"
   
    w_arr = np.fromfile(w_path, dtype='float32')
    b_arr = np.fromfile(b_path, dtype='float32')

    w_shape = layer.get_weights()[0].shape
    b_shape = layer.get_weights()[1].shape
    
    if "conv" in layer_name:      
      w_nchw_shape = (w_shape[3], w_shape[2], w_shape[0], w_shape[1])      
      w_arr = np.reshape(w_arr, w_nchw_shape)
      b_arr = np.reshape(b_arr, b_shape)
    
      w_arr = np.transpose(w_arr, (2,3,1,0))
      DEBUG ("old_shape = ", w_shape, " new_shape = ", w_arr.shape)

    if "dense" in layer_name:      
      w_arr = np.reshape(w_arr, w_shape)
      b_arr = np.reshape(b_arr, b_shape)
    
    weights = []
    weights.append(w_arr)
    weights.append(b_arr)
    # NOTE: overriding weights
    layer.set_weights(weights)

  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss2:', score[0])
  print('Test accuracy2:', score[1])
 

def getUniquePath(weights_dir):

  # Do not overwrite existing directories - create new with unique ID
  if os.path.exists(weights_dir):
    char_count = len(weights_dir)
    if weights_dir[char_count - 1] == "/":
      weights_dir = weights_dir[:char_count-1]
    
    tokens = weights_dir.split("_")
    last_tok = tokens[len(tokens) - 1]
    if last_tok.isdigit():
      id = int(last_tok)
      id += 1
      weights_dir = "_".join(tokens[:-1]) + "_" + str(id) + "/"
    else:
      weights_dir = "_".join(tokens) + "_1/"

    weights_dir = getUniquePath(weights_dir)
      
  
  return weights_dir
  


def createRecursiveDir(target_dir):

  if os.path.exists(target_dir):
    print ("Directory = ", target_dir, " exists ")
    print ("Delete Directory or Give Different Path. Aborting....")
    sys.exit(1)

  toks = target_dir.split("/")
  for i in range(len(toks)):
    path_str = "/".join(toks[0:i+1])
    if path_str != "":
      if not os.path.exists(path_str):
        os.mkdir(path_str)
  


#***** Top level External Function ******* 
def translate_to_approxhpvm(model,
                            weights_dir, src_dir,
                            test_data, test_labels,
                            tuner_data, tuner_labels,
                            batch_size, num_classes=10,
                            enable_weights_reload = False):


  reload_weights = enable_weights_reload   # If set to True, does not dump any weight/input/label files

  if not reload_weights:
    #weights_dir = getUniquePath(weights_dir)
    createRecursiveDir(weights_dir)
    

  #src_dir = getUniquePath(src_dir)
  createRecursiveDir(src_dir)
    
  dfg = DFG()    
  for i in range(len(model.layers)):
    layer = model.layers[i]
    # NOTE: Add DNN layer to graph
    dfg.add_to_graph(layer)

  # Build and Print DFG in reverse postorder
  dfg.buildDFG()


  DEBUG ("test_data.shape = ", test_data.shape, "\n")
  DEBUG ("test_labels.shape = ", test_labels.shape, "\n")

  tensorRtTranslator = TensorRtTranslator(dfg)
  tensorRtTranslator.setInputHeightWidth(test_data)
  tensorRtTranslator.translate(model, weights_dir, src_dir, test_data, test_labels, tuner_data, tuner_labels, reload_weights)
  weight_str = tensorRtTranslator.getWeightStr()
  input_str = tensorRtTranslator.getInputStr()


  filter_names = tensorRtTranslator.getFilterNames()
  hpvmTranslator = HPVMTranslator(dfg, weight_str, input_str, filter_names)    
  hpvmTranslator.translate(model, src_dir, test_data, tuner_data, batch_size)

  promiseTranslator = PromiseRtTranslator(dfg, weight_str)
  promiseTranslator.translate(model, src_dir, test_data)

  
  
  if reload_weights:
    print ("NOTE: Using existing pretrained weights \n")
  else:
    print ("NOTE: Translating Keras .h5 file to HPVM .bin files  \n")
    
  print ("-- Weight Files Under : ", weights_dir)
  print ("-- TensorRT src : ", src_dir + "/src.cc")
  print ("-- ApproxHPVM src  : ", src_dir + "approxhpvm_src.cc")

  
  return src_dir

