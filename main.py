# !paddle2onnx --model_dir '/home/rogday/Downloads/ch_ppocr_mobile_v2.0_rec_infer' --model_filename 'inference.pdmodel' --params_filename 'inference.pdiparams'  --save_file 'recognition.onnx' --opset_version 11 --enable_onnx_checker True


import onnx
import onnxsim
import numpy as np
from onnx import numpy_helper, helper, AttributeProto, TensorProto, GraphProto
import torch
from torch.autograd import Variable, Function
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
import copy

class TinyConv(nn.Module):

    def __init__(self):
        super(TinyConv, self).__init__()
        self.conv = nn.Conv2d(8, 8, 3, stride=[2, 1], padding=1, groups=8)


    def forward(self, x):
        return self.conv(x)


input = Variable(torch.randn(1, 8, 16, 50))
model = TinyConv()
model.eval()

# Export the model
torch.onnx.export(model,               # model being run
                  input,                         # model input (or a tuple for multiple inputs)
                  "tiny_conv.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                  #               'output' : {0 : 'batch_size'}}
                  )

model = onnx.load('tiny_conv.onnx')

def add_const(i, inp, name, dtype=np.int64):
    inp = np.array(inp, dtype=dtype)
    inp = onnx.numpy_helper.from_array(inp, name=name)
    node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=[name],
        value=inp
    )
    model.graph.node.insert(i, node)

def insert_slice(model, dst, inp_name, out_name, starts, ends, axes, steps):
    (starts_name, starts_data) = starts
    (ends_name, ends_data) = ends
    (axes_name, axes_data) = axes
    (steps_name, steps_data) = steps

    node = onnx.helper.make_node(
        'Slice',
        inputs=[inp_name, starts_name, ends_name, axes_name, steps_name],
        outputs=[out_name],
    )

    model.graph.node.insert(dst, node)

    add_const(dst, starts_data, starts_name)
    add_const(dst, ends_data, ends_name)
    add_const(dst, axes_data, axes_name)
    add_const(dst, steps_data, steps_name)    

def fix_convs(model):
    broken = []
    for i, node in enumerate(model.graph.node):
        if node.op_type == 'Conv':
            for at in node.attribute:
                if at.name == 'strides' and at.ints[0] != at.ints[1]:
                    assert len(at.ints) == 2
                    print(f"changing stride '{at.ints[0]}' in node '{node.name}' to match '{at.ints[1]}'")
                    at.ints[0] = at.ints[1]
                    broken.append(i)

    for i, idx in enumerate(broken):
        dst = idx + 5*i + 1

        old_output = model.graph.node[dst - 1].output[0]
        new_output = f'{old_output}_{dst}'
        model.graph.node[dst - 1].output[0] = new_output
        
        insert_slice(model, dst, inp_name=new_output, out_name=old_output, 
             starts=(f'starts_{dst}', [0]),
             ends=(f'ends_{dst}', [2**63 - 1]),
             axes=(f'axes_{dst}', [2]),
             steps=(f'steps_{dst}', [2]))

def fix_hardsigmoid(model):
    c_0 = onnx.helper.make_node(
        'Constant', [], ['Clip_constant_with_unique_name_0'],
        value=onnx.numpy_helper.from_array(np.array([0], dtype=np.float32), name='Clip_constant_with_unique_name_0'),
    )

    c_1 = onnx.helper.make_node(
        'Constant', [], ['Clip_constant_with_unique_name_6'],
        value=onnx.numpy_helper.from_array(np.array([6], dtype=np.float32), name='Clip_constant_with_unique_name_6'),
    )

    model.graph.node.insert(0, c_0)
    model.graph.node.insert(0, c_1)


    broken = []
    for i, node in enumerate(model.graph.node):
        if node.op_type == 'HardSigmoid':
            print(f'fixing HardSigmoid "{node.name}"')
            assert len(node.attribute) == 2 and node.attribute[0].name == 'alpha' and node.attribute[1].name == 'beta'
            broken.append(i)

    for i, idx in enumerate(broken):
        dst = idx + 6 * i # -1 + 7 = 6

        node = model.graph.node[dst]
        del model.graph.node[dst]

        alpha = node.attribute[0].f
        beta = node.attribute[1].f
        old_output = node.output[0]
        old_input = node.input[0]

        alpha_name = f'alpha_{dst}'
        beta_name = f'beta_{dst}'
        new_output = f'{old_output}_{dst}'

        node = onnx.helper.make_node(
            'Div',
            inputs=[new_output + "_clip6", "Clip_constant_with_unique_name_6"],
            outputs=[old_output],
        )

        model.graph.node.insert(dst, node)

        node = onnx.helper.make_node(
            'Clip',
            inputs=[new_output + "_mul", "Clip_constant_with_unique_name_0", "Clip_constant_with_unique_name_6"],
            outputs=[new_output + "_clip6"],
        )

        model.graph.node.insert(dst, node)

        node = onnx.helper.make_node(
            'Mul',
            inputs=[new_output + "_add", "Clip_constant_with_unique_name_6"],
            outputs=[new_output + "_mul"],
        )

        model.graph.node.insert(dst, node)

        node = onnx.helper.make_node(
            'Add',
            inputs=[new_output, beta_name],
            outputs=[new_output + "_add"],
        )

        model.graph.node.insert(dst, node)

        node = onnx.helper.make_node(
            'Mul',
            inputs=[old_input, alpha_name],
            outputs=[new_output],
        )

        model.graph.node.insert(dst, node)

        add_const(dst, [alpha], alpha_name, np.float32)
        add_const(dst, [beta], beta_name, np.float32)


def fix_tranpose(model):
    broken = []
    for i, node in enumerate(model.graph.node):
        if node.op_type == 'Transpose':
            assert len(node.attribute) == 1 and node.attribute[0].name == 'perm'
            if len(node.attribute[0].ints) != 4:
                broken.append(i)

    for i, idx in enumerate(broken):
        dst = idx + 2*i
        node = model.graph.node[dst]

        # fix Transpose node
        ints = list(node.attribute[0].ints)
        ints.insert(0, -1)
        ints = np.array(ints, dtype=np.int32) + 1
        del node.attribute[0]
        node.attribute.append(onnx.helper.make_attribute('perm', ints))

        old_input = node.input[0]
        new_input = f'{old_input}_{dst}'
        node.input[0] = new_input

        old_output = node.output[0]
        new_output = f'{old_output}_{dst}'
        node.output[0] = new_output

        node = onnx.helper.make_node(
            'Unsqueeze',
            inputs=[old_input],
            outputs=[new_input],
            axes=[0],
        )

        model.graph.node.insert(dst, node)

        node = onnx.helper.make_node(
            'Squeeze',
            inputs=[new_output],
            outputs=[old_output],
            axes=[0],
        )

        model.graph.node.insert(dst + 2, node)
    
def check_save(model, name):
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, name)

def simplify(name, rename=False, **kwargs):
    model, check = onnxsim.simplify(name, **kwargs)
    assert check, "couldn't valide"
    name = name[:-5]
    if rename:
        name += '_optimized'
    onnx.save(model, name + '.onnx')

name = 'recognition'
simplify(name + ".onnx", input_shapes={'x' : [1, 3, 32, 100]})

model = onnx.load(name + '.onnx')

def insert_const(model, name, data):
    C_0 = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, list(data.shape))
    c_0 = onnx.helper.make_node(
        'Constant', [], [name],
        value=onnx.numpy_helper.from_array(data, name=name),
    )
    model.graph.node.insert(0, c_0)
    model.graph.value_info.insert(0, C_0)

def fix_constants(model):
    # find init and input indices
    
#     for i, node in enumerate(model.graph.input):
#         if node.name == 'Constant_0' or node.name == 'Constant_2':
#             print(i, node.name)

    # name        val  init input
    # Constant_0: 6.0, #20  #21
    # Constant_2: 0.0, #22  #23

    del model.graph.initializer[22]
    del model.graph.initializer[20]

    del model.graph.input[23]
    del model.graph.input[21]


    insert_const(model, 'Constant_0', np.array([6], dtype=np.float32))
    insert_const(model, 'Constant_2', np.array([0], dtype=np.float32))

fix_constants(model)
fix_convs(model)
# del model.graph.output[0]
# model.graph.output.append(onnx.helper.make_tensor_value_info('save_infer_model/scale_0.tmp_1', onnx.TensorProto.FLOAT, [1, 2]))

del model.graph.output[0]
model.graph.output.append(onnx.helper.make_tensor_value_info('save_infer_model/scale_0.tmp_1', onnx.TensorProto.FLOAT, [1, 25, 6625]))


def fix_lstm(model):
    broken = []
    inputs = set()

    for i, node in enumerate(model.graph.node):
        if node.op_type == 'LSTM':
            broken.append(i)
            for j, in_node in enumerate(model.graph.initializer):
                if in_node.name in node.input:
                    inputs.add(j)
    
    for idx in inputs:
        name = model.graph.initializer[idx].name
        for node in model.graph.input:
            if node.name == name:
                model.graph.input.remove(node)
                break

    for i, idx in enumerate(sorted(inputs)): # hashset by deafult?
        dst = idx + i # -1 + 2 = 1

        inp = model.graph.initializer[dst]

        data = onnx.numpy_helper.to_array(inp)
        [data_f, data_b] = np.split(data, [1])

        data_f = onnx.numpy_helper.from_array(data_f, name=inp.name + "_f")
        data_b = onnx.numpy_helper.from_array(data_b, name=inp.name + "_b")

        del model.graph.initializer[dst]

        model.graph.initializer.insert(dst, data_f)
#         print('inserted: ', model.graph.initializer[dst].name)
        model.graph.initializer.insert(dst, data_b)
#         print('inserted: ', model.graph.initializer[dst].name)

    def change_input(node, suffix):
        new_input = [x + suffix for x in node.input[1:]]
        new_input.insert(0, node.input[0])
        new_input[4] = ''
        
        for x in range(7):
            del node.input[0]
        
        for name in new_input:
            node.input.append(name)

    for i, idx in enumerate(broken):
        dst = idx + 12*i # 0 + 7
        
        node_f = model.graph.node[dst]
        
        assert node_f.attribute[0].name == 'direction'
        del node_f.attribute[0]
        node_f.output[1] = ''
        node_f.output[2] = ''
        node_f.attribute.append(onnx.helper.make_attribute('direction', 'forward'))
        
        node_b = copy.deepcopy(node_f)
        node_b.name += '_b'
        change_input(node_f, '_f')
        change_input(node_b, '_b')
        
        old_input = node_f.input[0]
        new_input = f'{old_input}_{dst}'
        node_b.input[0] = new_input
        
        old_output = node_f.output[0]
        new_output_f = f'{old_output}_{dst}_f'
        new_output_b = f'{old_output}_{dst}_b'
        node_f.output[0] = new_output_f
        node_b.output[0] = new_output_b
        
        dst += 1
        
        node = onnx.helper.make_node(
            'Concat',
            inputs=[new_output_f, new_output_b + '_rev'],
            outputs=[old_output],
            axis=1
        )
        model.graph.node.insert(dst, node)
        
        ### insert reverse
        insert_slice(model, dst, inp_name=new_output_b, out_name=new_output_b + '_rev', 
                     starts=(f'starts_lstm_{dst}_out', [-1]),
                     ends=(f'ends_lstm_{dst}_out', [-2**63 + 1]),
                     axes=(f'axes_lstm_{dst}_out', [0]),
                     steps=(f'steps_lstm_{dst}_out', [-1]))
        
        ###
        
        model.graph.node.insert(dst, node_b)
        
        insert_slice(model, dst, inp_name=old_input, out_name=new_input, 
             starts=(f'starts_lstm_{dst}', [-1]),
             ends=(f'ends_lstm_{dst}', [-2**63 + 1]),
             axes=(f'axes_lstm_{dst}', [0]),
             steps=(f'steps_lstm_{dst}', [-1]))

fix_hardsigmoid(model)
fix_tranpose(model)
fix_lstm(model)
check_save(model, name + '_fixed.onnx')
print('ok')


class LSTM(nn.Module):
    def __init__(self, features, hidden, batch, num_layers=1, bidirectional=False):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(features, hidden, num_layers, bidirectional=bidirectional)
        self.h0 = torch.zeros(num_layers + int(bidirectional), batch, hidden)
        self.c0 = torch.zeros(num_layers + int(bidirectional), batch, hidden)

    def forward(self, x):
        return self.lstm(x, (self.h0, self.c0))[0] #, torch.flip(x, [0])

batch = 5
features = 4
hidden = 3
seq_len = 2

input = Variable(torch.randn(seq_len, batch, features))
model = LSTM(features, hidden, batch, bidirectional=True)
model.eval()

# Export the model
torch.onnx.export(model,               # model being run
                  input,                         # model input (or a tuple for multiple inputs)
                  "tiny_lstm.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                  #               'output' : {0 : 'batch_size'}}
                  )

simplify('tiny_lstm.onnx')
model = onnx.load('tiny_lstm.onnx')
fix_lstm(model)
check_save(model, 'tiny_lstm_fixed.onnx')

import onnxruntime as ort
x = np.random.rand(seq_len, batch, features).astype(np.float32)

ort_sess = ort.InferenceSession('tiny_lstm.onnx')
y_ref = ort_sess.run(None, {'input': x})[0]

ort_sess = ort.InferenceSession('tiny_lstm_fixed.onnx')
y = ort_sess.run(None, {'input': x})[0]

print(np.allclose(y_ref, y))
