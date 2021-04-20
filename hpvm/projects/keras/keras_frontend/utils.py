


def DEBUG(str, *args):

  debug = False
  if debug:
    print (str, *args)



def nodeHasBias(cur_node):
    
  if cur_node.layer_type == "Conv2D" or cur_node.layer_type == "DepthwiseConv2D" or cur_node.layer_type == "Dense":
    #return True
    return cur_node.use_bias
  else:
    return False

  
def layerHasActivationAttr(cur_node):
    
  if cur_node.layer_type == "Conv2D" or cur_node.layer_type == "DepthwiseConv2D" \
     or cur_node.layer_type == "Dense" or cur_node.layer_type == "Activation":
    return True
  else:
    return False


def nodeHasActivation(cur_node):
    
  if cur_node.layer_type == "Conv2D" or cur_node.layer_type == "DepthwiseConv2D" \
     or cur_node.layer_type == "Dense" or cur_node.layer_type == "Activation":
    #return True
    return cur_node.activation_type != "linear"
  else:
    return False


def genActivationCallStr(input_var, output_var, activation_type):
 
  func_name = ""
  if activation_type == "tanh":
    func_name = "Tanh"

  if activation_type == "relu":
    func_name = "Relu"

  if activation_type == "softmax":
    func_name = "Softmax"

  inst_str = "void* " + output_var + " = "
  inst_str += "tensor" + func_name + "(" + input_var + "); \n"

    
  return inst_str

  
