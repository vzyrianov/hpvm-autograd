

import random
import numpy as np
import sys
from keras import backend as K
from keras_frontend.utils import *
from keras_frontend.quantize_utils import get_best_quant_range, dumpQuantizeRanges


class State:

  def __init__(self):
    self.ops = []
    self.op_string = ""
    self.num_ops = 0 

  def clear(self):
    self.ops = []
    self.op_string = ""
    self.num_ops = 0

  def add(self, cur_node, layer_type):
    self.ops.append(cur_node)
    if self.op_string != "":
        self.op_string += "_"
        
    self.op_string += layer_type
    self.num_ops += 1
    
    
  def getFirstOp(self):
    return self.ops[0]

  def getLastOp(self):
    return self.ops[self.num_ops - 1]

  def getDenseOp(self):
    for i in range(self.num_ops):
      if self.ops[i].layer_type == "Dense":
        return self.ops[i] 
    return None  # Should be unreachable  

  
  def getConvOp(self):
    for i in range(self.num_ops):
      if self.ops[i].layer_type == "Conv2D" or self.ops[i].layer_type == "DepthwiseConv2D":
        return self.ops[i] 
    return None  # Should be unreachable  

    
  def getDepthwiseConvOp(self):
    for i in range(self.num_ops):
      if self.ops[i].layer_type == "DepthwiseConv2D":
        return self.ops[i] 
    return None  # Should be unreachable  


  def getPoolOp(self):
    for i in range(self.num_ops):
      layer_type = self.ops[i].layer_type
      if layer_type  == "MaxPooling2D" or layer_type == "AveragePooling2D":
        return self.ops[i] 
    return None  # Should be unreachable  

  
  def getPadOp(self):
    for i in range(self.num_ops):
      layer_type = self.ops[i].layer_type
      if layer_type  == "ZeroPadding2D":
        return self.ops[i] 
    return None  # Should be unreachable  

  
  def getActivationID(self):    
    activation_type = 'linear'
    for i in range(self.num_ops):
      cur_op = self.ops[i]
      layer_type = cur_op.layer_type
      if layer_type == "Dense" or layer_type == "Conv2D" or layer_type == "Activation":
        activation_type = cur_op.activation_type

    activation_id = -1;
    if activation_type == "tanh":
      activation_id = 0
    if activation_type == "relu":
      activation_id = 1
      
    return activation_id 
  
  
  def isDense(self):
    if "dense" in self.op_string:
      return True
    return False

  def isConv(self):
    if "conv" in self.op_string:
      return True
    return False

  def isDepthwiseConv(self):
    if "depthwise" in self.op_string:
      return True
    return False

  
  def isBatchNorm(self):
    if "batchnorm" in self.op_string:
      return True
    return False


  def isPool(self):
    if "pool" in self.op_string and self.num_ops == 1:
      return True
    return False

  
  def isActivation(self):
    if "activation" in self.op_string and self.num_ops == 1:
      return True
    return False
  

  def getPadding(self):    
    padding_op = self.getPadOp()
    prev_padding = 0 
    if padding_op is not None:
      prev_padding = padding_op.padding[0][0]
    
    conv_op = self.getConvOp()
    if conv_op is None:
      print ("ERROR: Conv Op not found")
      sys.exit(0)

    K = conv_op.weights.shape[0]
    padding = 0
    padding_type = conv_op.padding.strip()
    if padding_type == "valid":
      padding = 0
    else:
      padding = int((K - 1) / 2)

    padding = padding + prev_padding
    
    return padding  
               
      
  def getPoolInfo(self):    
    pool_op =  self.getPoolOp()
    if pool_op is None:
      return -1, [0,0]

    pool_id = -1
    layer_type = pool_op.layer_type
    if layer_type == "MaxPooling2D":
      pool_id = 0
    if layer_type == "AveragePooling2D":
      pool_id = 1

    pool_size = pool_op.pool_size    
    return pool_id, pool_size
  
    
  def getStrides(self):    
    conv_op = self.getConvOp()
    strides = conv_op.strides
    return strides
  
      


class PromiseRtTranslator:

  # NOTE: weight_str can be optinally passed
  def __init__(self, dfg, weight_str = ""):
    self.dfg = dfg
    self.output_map = {}
    self.visited_nodes = {}
    self.counter = 0
    self.weight_str = weight_str
    self.program_str = ""    
    self.swing_value = 9  # FP32
    self.quant_ranges = {}
    # Used to generate PromiseSim Info
    self.layer_str = ""
    self.cur_layer_id = 1
    self.layer_size_str = "" 
    self.layer_input_sizes = {}
    self.unique_op_types = {}
    self.batch_size = 0
    self.weights_dir = ""

    
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


  def isForwardLayer(self, layer_type):

    skip_layers = {}
    skip_layers["Input"] = 0
    skip_layers["InputLayer"] = 0
    skip_layers["Flatten"] = 0
    skip_layers["Dropout"] = 0
    if layer_type in skip_layers:
      return True
    else:
      return False


    
  def appendLayerSizeStr(self, promise_layer_type, state):

    central_op = None
    if promise_layer_type == "Conv":
      central_op = state.getConvOp()
    if promise_layer_type == "FC":
      central_op = state.getDenseOp()
      
    first_op = state.getFirstOp()
    layer_name = first_op.layer_name
        
    unique_id = 0
    if promise_layer_type not in self.unique_op_types:
      self.unique_op_types[promise_layer_type] = 1
      unique_id = 1
    else:
      unique_id = self.unique_op_types[promise_layer_type]
      unique_id += 1
      self.unique_op_types[promise_layer_type] = unique_id

    unique_layer_name = promise_layer_type + str(unique_id)
    if promise_layer_type == "Conv" or promise_layer_type == "FC":
      self.layer_size_str += unique_layer_name + ","
    else:
      # Handling single tensor ops - NO Promise layer
      self.layer_size_str += "#tensor" + unique_layer_name + "\n"
      return 

    
    weights_shape = central_op.weights.shape
    input_size = self.layer_input_sizes[layer_name]  
    N = self.batch_size
    C = input_size[1]

    if str(C) == "?":
      C = weights_shape[0]
   
    self.layer_size_str += str(N) + "," + str(C) + ","

    if promise_layer_type == "Conv":
      H = input_size[2]
      W = input_size[3]
      self.layer_size_str += str(H) + "," + str(W) + "," 
    
  
    H = weights_shape[0]
    W = weights_shape[1]

    if promise_layer_type == "Conv":
      N = weights_shape[3]
      C = weights_shape[2]
      self.layer_size_str += str(N) + "," + str(C) + ","

    self.layer_size_str += str(H) + "," + str(W) 
      
    self.layer_size_str += "\n"
    
        
    
    

  def appendLayerString(self, promise_layer_type, state):

    
    layer_str = str(self.cur_layer_id) + " gpu "
    self.cur_layer_id += 1
    
    for op in state.ops:
      op_type = op.layer_type    
      if op_type == "Conv2D":
        layer_str += "conv fp32 1 "
        if op.use_bias:
          layer_str += "add fp32 1 "
        if op.activation_type != "linear":
          layer_str += op.activation_type + " fp32 1 "

      if op_type == "DepthwiseConv2D":
        layer_str += "group_conv fp32 1"
        if op.use_bias:
          layer_str += "add "
        if op.activation_type != "linear":
          layer_str += op.activation_type + " fp32 1"

      if op_type == "BatchNormalization":
        layer_str += "batchnorm fp32 1 "
          
      if op_type == "Dense":
        layer_str += "mul fp32 1 "
        if op.use_bias:
          layer_str += "add fp32 1 "
        if op.activation_type != "linear":
          layer_str += op.activation_type + " fp32 1 "

          
      if op_type == "MaxPooling2D":
        layer_str += "pool_max fp32 1 "

      if op_type == "AveragePooling2D":
        layer_str += "pool_mean fp32 1 "
      
      if op_type == "Add":    
        layer_str += "add fp32 1 "

      if op_type == "Activation":
        layer_str += op.activation_type + " fp32 1 "

    layer_str += "\n"

    self.layer_str += layer_str
    
    self.appendLayerSizeStr(promise_layer_type, state)    
    
        

    
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

  
  # Retrieve input name of the previous layer
  def getInputLayerName(self, cur_node):

    # Assumption: If no inputs, the previous layer must be input layer
    if len(cur_node.inputs) == 0:
      return "input"

    pred_layer_type = cur_node.inputs[0].layer_type
    # FIXME: Assuming the 'inference' phase - hence skipping Dropout
    #if pred_layer_type == "Flatten" or pred_layer_type == "Dropout":
    if self.isSkipLayer(pred_layer_type):
      cur_node = self.getPrevActiveLayer(cur_node)

    if cur_node.inputs[0].layer_type == "InputLayer":
      return "input"
  
    # get input to the layer
    input_node_name = cur_node.inputs[0].layer_name  # get the input layer ID

    return input_node_name
  
      
    
  # Retrieve input name of the previous layer
  def getSingleInputName(self, cur_node):

    # Assumption: If no inputs, the previous layer must be input layer
    if len(cur_node.inputs) == 0:
      return "input"

    pred_layer_type = cur_node.inputs[0].layer_type
    # FIXME: Assuming the 'inference' phase - hence skipping Dropout
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
      print ("Input Var not found - Aborting....", input_node_name, "\n")
      sys.exit(0)
      
    return input_var_name



  # Used to retrieve inputs for "add" operation with multiple inputs  
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
  
  

  def getWeightNames(self, cur_node):
    
    layer_name = cur_node.layer_name
    w_name = layer_name + "_w"
    b_name = layer_name + "_b"
    # If Conv has no bias Add operation
    if cur_node.use_bias == False:
      b_name = "NULL"
    
    return w_name, b_name


  def getWeightRange(self, cur_node):

    layer_type = cur_node.layer_type
    if layer_type != "Dense" and layer_type != "Conv2D":
      print ("ERROR: layer_type = ", layer_type , " does not have weights ")
      sys.exit(0)

    weights = cur_node.weights

    (min_val, max_val) = get_best_quant_range(weights)
    
    
    return min_val, max_val

  
  def getBiasRange(self, cur_node):

    layer_type = cur_node.layer_type
    if layer_type != "Dense" and layer_type != "Conv2D":
      print ("ERROR: layer_type = ", layer_type , " does not have weights ")
      sys.exit(0)

    if cur_node.use_bias == False:
      return 0, 0
    
    bias_weights = cur_node.bias_weights
    min_val = np.amin(bias_weights)
    max_val = np.amax(bias_weights)
    
    return min_val, max_val


  # Returns the output value ranges for the input and output to a PROMISE layer
  def getQuantRange(self, state):
    
    first_op = state.getFirstOp()
    last_op = state.getLastOp()

    prev_layer_name = self.getInputLayerName(first_op)
    cur_layer_name = last_op.layer_name

    if prev_layer_name not in self.quant_ranges or cur_layer_name not in self.quant_ranges:
      print ("ERROR: Layer_name = ", prev_layer_name ," or ", cur_layer_name, " not found in quant_range")
      sys.exit(0)
      
    input_quant_range = self.quant_ranges[prev_layer_name]
    output_quant_range = self.quant_ranges[cur_layer_name]

    print (input_quant_range)
    print (output_quant_range)
    
    return input_quant_range, output_quant_range

  
    
  def genDenseLayer(self, state):
    
    first_op = state.getFirstOp()
    dense_op = state.getDenseOp()
    last_op = state.getLastOp()

    input_var = self.getSingleInputName(first_op)
    output_var = self.getVariableName(last_op)

    w_name, b_name = self.getWeightNames(dense_op)
    w_min, w_max = self.getWeightRange(dense_op)
    b_min, b_max = self.getBiasRange(dense_op)   
    
    activation_id = state.getActivationID()
    
    self.appendLayerString("FC", state)
    
    state.clear()



    
  def genConvLayer(self, state):
    
    first_op = state.getFirstOp()
    conv_op = state.getConvOp()
    last_op = state.getLastOp()

    input_var = self.getSingleInputName(first_op)
    output_var = self.getVariableName(last_op)

    w_name, b_name = self.getWeightNames(conv_op)
    w_min, w_max = self.getWeightRange(conv_op)
    b_min, b_max = self.getBiasRange(conv_op)   
    
    activation_id = state.getActivationID()
    padding = state.getPadding()
    pool_id, pool_size = state.getPoolInfo()
    strides = state.getStrides()

    self.appendLayerString("Conv", state)

    state.clear()

    


  def genDepthwiseConvLayer(self, state):
  
    conv_op = state.getDepthwiseConvOp()
    first_op = state.getFirstOp()
    last_op = state.getLastOp()

    input_var = self.getSingleInputName(first_op)
    output_var = self.getVariableName(last_op)

    w_name, b_name = self.getWeightNames(conv_op)

    activation_id = state.getActivationID()
    padding = state.getPadding()
    pool_id, pool_size = state.getPoolInfo()
    strides = state.getStrides()
    
    self.appendLayerString("DepthwiseConv", state)

    state.clear()



  def genBatchNormLayer(self, state):

    first_op = state.getFirstOp()
    last_op = state.getFirstOp()

    input_var = self.getSingleInputName(first_op)
    output_var = self.getVariableName(last_op)

    self.appendLayerString("BatchNorm", state)

    state.clear()

    
    

  def genSoftmaxLayer(self, state):
  
    first_op = state.getFirstOp()
    last_op = state.getLastOp()

    self.layer_str += str(self.cur_layer_id) + " gpu softmax fp32 1\n"  
    
    state.clear()


  def genAddLayer(self, state):
  
    first_op = state.getFirstOp()
    last_op = state.getLastOp()

    input_vars = self.getMultipleInputNames(first_op)
    output_var = self.getVariableName(last_op)
    
    promise_layer_str = "void* " + output_var + " = tensorAdd(" + input_vars[0]
    promise_layer_str += ", " + input_vars[1] + "); \n"
    #print (promise_layer_str)

    self.program_str += promise_layer_str


    self.appendLayerString("Add", state)

    state.clear()


    
    
  def genActivationLayer(self, state):
  
    first_op = state.getFirstOp()
    input_var = self.getSingleInputName(first_op)
    output_var = self.getVariableName(first_op)

    activation_type = first_op.activation_type
    
    func_name = ""
    if activation_type == "tanh":
      func_name = "Tanh"

    if activation_type == "relu":
      func_name = "Relu"

    inst_str = "void* " + output_var + " = "
    inst_str += "tensor" + func_name + "(" + input_var + "); \n"

    self.program_str += inst_str


    self.appendLayerString(func_name, state)

    state.clear()

    
  # FIXME: Only supporting single AveragePooling layers
  def genPoolLayer(self, state):
  
    # For single pool layer should be all same
    pool_op = state.getPoolOp()

    input_var = self.getSingleInputName(pool_op)
    output_var = self.getVariableName(pool_op)
   
    pool_size = pool_op.pool_size
    strides = pool_op.strides
    # FIXME: Same padding is *NOT* currently supported
    padding = 0
    pool_type = 0

    layer_type = pool_op.layer_type
    if layer_type == "MaxPooling2D":
      pool_type = "0"
    if layer_type == "AveragePooling2D":
      pool_type = "1"     
      
    # tensorPooling(input, pool_type, pool_h, pool_w, v_pad, h_pad, v_stride, h_stride)
    inst_str = "void* " + output_var + " = "
    inst_str += "tensorPooling(" + input_var + "," + pool_type + "," + str(pool_size[0]) + "," + str(pool_size[1]) 
    inst_str +=  "," + str(padding) + "," + str(padding) + "," + str(strides[0]) + "," + str(strides[1])
    inst_str += "); \n"
    self.program_str += inst_str


    self.appendLayerString("Pooling", state)

    state.clear()



  

  def genPreviousLayer(self, state):

    if state.isDense():
      self.genDenseLayer(state)
      
    elif state.isConv():
      self.genConvLayer(state)

    elif state.isDepthwiseConv():
      self.genDepthwiseConvLayer(state)
      
    elif state.isBatchNorm():
      self.genBatchNormLayer(state)
      
    elif state.isPool():
      self.genPoolLayer(state)

    elif state.isActivation():
      self.genActivationLayer(state)



      
  def shouldVisit(self, cur_node):
    layer_name = cur_node.layer_name
    # NOTE: visit a node if not already visited and all predecessors are visited
    if self.dfg.predVisited(cur_node, self.visited_nodes) and layer_name not in self.visited_nodes:
      return True
    else:
      return False
      
    
  def handle_padding(self, cur_node, state):
    if not self.shouldVisit(cur_node):
      return  

    layer_name = cur_node.layer_name
    self.visited_nodes[layer_name] = True

    self.genPreviousLayer(state)

    # Appending conv to state
    state.add(cur_node, "padding")
   
    self.traverseSuccessors(cur_node, state)
  

    
  def handle_conv(self, cur_node, state):
    if not self.shouldVisit(cur_node):
      return  

    layer_name = cur_node.layer_name
    self.visited_nodes[layer_name] = True

    self.genPreviousLayer(state)

    # Appending conv to state
    state.add(cur_node, "conv")
    
    self.traverseSuccessors(cur_node, state)


    
  def handle_depthwise_conv(self, cur_node, state):
    if not self.shouldVisit(cur_node):
      return  

    layer_name = cur_node.layer_name
    self.visited_nodes[layer_name] = True

    self.genPreviousLayer(state)

    # Appending depthwise_conv to state
    state.add(cur_node, "depthwise")
    
    self.traverseSuccessors(cur_node, state)



  def handle_batchnorm(self, cur_node, state):
    if not self.shouldVisit(cur_node):
      return  

    layer_name = cur_node.layer_name
    #print ("handle_batchnorm", layer_name)
    self.visited_nodes[layer_name] = True

    self.genPreviousLayer(state)

    state.add(cur_node, "batchnorm")
    
    self.genBatchNormLayer(state)    
    
    self.traverseSuccessors(cur_node, state)



    
    
  def handle_add(self, cur_node, state):
    if not self.shouldVisit(cur_node):
      return

    layer_name = cur_node.layer_name
    self.visited_nodes[layer_name] = True

    self.genPreviousLayer(state)

    # Appending conv to state
    state.add(cur_node, "add")

    self.genAddLayer(state)    
    
    self.traverseSuccessors(cur_node, state)


      
  def handle_activation(self, cur_node, state):
    if not self.shouldVisit(cur_node):
      return

    layer_name = cur_node.layer_name
    self.visited_nodes[layer_name] = True

    # NOTE: If end of DNN
    if cur_node.activation_type == "softmax":
      self.genPreviousLayer(state)
      state.add(cur_node, "activation")
      self.genSoftmaxLayer(state)
      # NOTE: return when observed end of DNN (softmax)
      return
    
    # Appending activation to state
    state.add(cur_node, "activation")

    self.traverseSuccessors(cur_node, state)

    
  def handle_dense(self, cur_node, state):
    if not self.shouldVisit(cur_node):
      return

    layer_name = cur_node.layer_name
    self.visited_nodes[layer_name] = True

    self.genPreviousLayer(state)

    state.add(cur_node, "dense")

    self.traverseSuccessors(cur_node, state)


  def handle_pooling(self, cur_node, state):    
    if not self.shouldVisit(cur_node):  
      return
  
    layer_name = cur_node.layer_name
    self.visited_nodes[layer_name] = True

    layer_type = cur_node.layer_type
    if layer_type == "AveragePooling2D":
      self.genPreviousLayer(state)
  
    state.add(cur_node, "pool")

    # NOTE: Will only generate pool layer if it is a standalone Pool (with no convolution)
    # self.genPreviousLayer(state)
    
    self.traverseSuccessors(cur_node, state)


  

  
  def handleLayers(self, output_node, state):

    layer_type = output_node.layer_type

    if layer_type == "ZeroPadding2D":
      self.handle_padding(output_node, state)

    if layer_type == "Conv2D":
      self.handle_conv(output_node, state)

    if layer_type == "DepthwiseConv2D":
      self.handle_depthwise_conv(output_node, state)

    if layer_type == "BatchNormalization":
      self.handle_batchnorm(output_node, state)

    if layer_type == "Dense":
      self.handle_dense(output_node, state)

    if layer_type == "Activation":
      self.handle_activation(output_node, state)

    if layer_type == "MaxPooling2D" or layer_type == "AveragePooling2D":
      self.handle_pooling(output_node, state)

    if layer_type == "Add":
      self.handle_add(output_node, state)
        
    if(self.isForwardLayer(layer_type)):
      layer_name = output_node.layer_name
      #print ("NOTE: Skippping = ", layer_name)
      self.visited_nodes[layer_name] = True
      self.traverseSuccessors(output_node, state)   


    
  def traverseSuccessors(self, cur_node, state):

    for output_node in cur_node.outputs:
      self.handleLayers(output_node, state)



      
  def findQuantizeRanges(self, model, x_test):

    inp = model.input                                           # input placeholder
    
    outputs = [layer.output for layer in model.layers]          # all layer outputs
    functor = K.function([inp, K.learning_phase()], outputs )   # evaluation function


    layer_ranges = {} 
    layer_it = 0
    for layer in model.layers:
      layer_name = model.layers[layer_it].name
      layer_ranges[layer_name] = []
      layer_it += 1

    batch_size = 1000
    input_size = len(x_test)
    num_batches = input_size // batch_size 

    i = 0
    while 1:

      import gc
      gc.collect()
      
      start = i * batch_size
      end = (i + 1) * batch_size
      
      # Inference over test set
      layer_outs = functor([x_test[start:end], 1.])
    
      # NOTE: Saving quant ranges for input
      min_val = np.amin(x_test)
      max_val = np.amax(x_test)
      self.quant_ranges["input"] = (min_val, max_val)          
    
      ind = 0
      for layer_out in layer_outs:
        layer_name = model.layers[ind].name
    
        (min_val, max_val) = get_best_quant_range(layer_out)    
        #print ("min_val = ", min_val, " max_val = ", max_val)

        layer_ranges[layer_name].append((min_val, max_val))
        #self.quant_ranges[layer_name] = (min_val, max_val)
        ind += 1 

      i += 1
      if i >= num_batches:
        break

      
        
    ind = 0
    for layer in model.layers:
      layer_name = model.layers[ind].name
      
      ranges = layer_ranges[layer_name]
      min_val = ranges[0][0]
      max_val = ranges[0][1]
      
      for range in ranges:
        if range[0] < min_val:
          min_val = range[0]
        if range[1] > max_val:
          max_val = range[1]
        
      self.quant_ranges[layer_name] = (min_val, max_val)    

      #print ("---- min = ", min_val, "  max = ", max_val, " ----- \n\n")

      ind += 1


      

      
  def findLayerInputSizes(self, model, x_test):

    self.batch_size = len(x_test)  
    for layer in model.layers:
      layer_type = layer.__class__.__name__ 
      if layer_type == "InputLayer" or layer_type == "Add":
        continue

      layer_name = layer.name
      #print ("layer_name = ", layer_name)
      #print ("layer_shape = ", layer.input.shape)
      self.layer_input_sizes[layer_name] = layer.input.shape


    

  def genExecutionLoop(self):

    exec_loop = ""
    exec_loop += "int total_runs = 100; \n"
    exec_loop += "for (int i = 0 ; i < total_runs; i++){ \n\n"

    return exec_loop
    

  def endExecutionLoop(self):

    end_exec_loop = "\n}\n"  
    return end_exec_loop
    

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
    loop_str += "\n\n" + self.weight_str + "\n\n"
    loop_str += "int start = i * batch_size; \n"
    loop_str += "int end = (i + 1) * batch_size; \n"

    loop_str += "\nvoid* input = readInputBatch(input_path.c_str(),0,start,end," 
    loop_str += str(C) + "," + str(H) + "," + str(W) + "); \n\n"

    return loop_str


    
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

    return end_loop_str

      

      
  def genHeader(self):

    headers = "\n#include <stdio.h> \n"
    headers += "#include <stdlib.h> \n"
    headers += "#include <unistd.h> \n"
    headers += "#include <fcntl.h> \n"
    headers += "#include <sys/types.h> \n"
    headers += "#include <sys/stat.h> \n"
    headers += "#include <string.h> \n"

    headers += "#include \"../../../tensor_runtime/include/tensor_runtime.h\" \n"
    headers += "#include \"../../include/utils.h\" \n\n"

    main_func = "int main(){ \n\n"

    initialization = "llvm_hpvm_initTensorRt(0); \n\n"

    # Merging into one header string
    header_str = headers
    header_str += main_func
    header_str += initialization

    return header_str
  
  

    
  def genFooter(self, test_data):

    footer_str = ""    
    if test_data is not None and self.dfg.last_node is not None:
      last_node = self.dfg.last_node
      output_var = self.output_map[last_node.layer_name]
 
    accuracy_call =  "\ndumpExecutionAccuracies(); \n"
    footer_str += accuracy_call
    
    destructors = "\nllvm_hpvm_cleanupTensorRt(); \n"
    footer_str += destructors
    
    end_main = "\nreturn 0; \n\n}\n"
    footer_str += end_main

    return footer_str
    


  def dumpLayerStr(self, dir_prefix):

    config_str = "0\n"
    config_str += "+++++\n"
    config_str += "conf1 1 1 100 0\n"
    config_str += self.layer_str
    config_str += "-----"

    f = open(dir_prefix + "/tuner_confs.txt", "w+")
    f.write(config_str)
    f.close()

    #f = open(dir_prefix + "/layers.txt", "w+")
    #f.write(self.layer_size_str)
    #f.close()

    
      
  def dumpProgramString(self, final_str, dir_prefix):

    f = open(dir_prefix + "/promise_src.cc", "w+")
    f.write(final_str)
    f.close()


    
  def generateSourceProgram(self, weights_dir, x_test):

    final_str = ""
    header_str = self.genHeader()
    final_str += header_str

    exec_loop = self.genExecutionLoop()
    final_str += exec_loop
    
    loop_str = self.genBatchLoop(x_test)
    final_str += loop_str
    
    final_str += self.program_str

    end_loop_str = self.endBatchLoop()
    final_str += end_loop_str

    end_exec_loop = self.endExecutionLoop()
    final_str += end_exec_loop

    footer_str = self.genFooter(x_test)
    final_str += footer_str    
    #print (final_str)
    
    self.dumpProgramString(final_str, weights_dir)
    
          
  
  
    
  def translate(self, model, weights_dir, x_test):

    #print ("\n\n\n **** PromiseRT Translator ****** \n\n\n")
    root_node = self.dfg.root_node
    state = State()

    self.weights_dir = weights_dir #  FIXIT: move this to class constructor
    
    self.findLayerInputSizes(model, x_test)
    
    self.handleLayers(root_node, state)

    # Commented out Promise code-gen - Not needed in this release version
    #self.generateSourceProgram(weights_dir, x_test)

    self.dumpLayerStr(weights_dir)

    
    

    
