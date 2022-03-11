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

def simplify(name, rename=False, **kwargs):
    model, check = onnxsim.simplify(name, **kwargs)
    assert check, "couldn't valide"
    name = name[:-5]
    if rename:
        name += '_optimized'
    onnx.save(model, name + '.onnx')

def check_save(model, name):
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, name)


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
torch.onnx.export(model,  # model being run
                  input,  # model input (or a tuple for multiple inputs)
                  "tiny_conv.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=11,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                  #               'output' : {0 : 'batch_size'}}
                  )

model = onnx.load('tiny_conv.onnx')


##############
class TinyLSTM(nn.Module):

    def __init__(self, features, hidden, batch, num_layers=1, bidirectional=False):
        super(TinyLSTM, self).__init__()
        self.lstm = nn.LSTM(features, hidden, num_layers, bidirectional=bidirectional, bias=True)
        self.h0 = torch.zeros(num_layers + int(bidirectional), batch, hidden)
        self.c0 = torch.zeros(num_layers + int(bidirectional), batch, hidden)

    def forward(self, x):
        return self.lstm(x, (self.h0, self.c0))[0]

batch = 1
features = 4
hidden = 3
seq_len = 2

input = Variable(torch.randn(seq_len, batch, features))
model = TinyLSTM(features, hidden, batch, bidirectional=False)
model.eval()

# Export the model
torch.onnx.export(model,  # model being run
                  input,  # model input (or a tuple for multiple inputs)
                  "tiny_lstm.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  # dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                  #               'output' : {0 : 'batch_size'}}
                  )

simplify('tiny_lstm.onnx')#, skipped_optimizers=['extract_constant_to_initializer'])
model = onnx.load('tiny_lstm.onnx')
print(model)
# assert False

# model.graph.node[0].input[5] = ''
# model.graph.node[0].input[6] = ''
#
# del model.graph.initializer[4]
# del model.graph.initializer[3]

def add_const_tiny(model, init_id):
    data = model.graph.initializer[init_id]
    del model.graph.initializer[init_id]
    shp = onnx.numpy_helper.to_array(data).shape
    name = data.name
    value_info = onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, list(shp))
    node = onnx.helper.make_node(
        'Constant', [], [name],
        value=data,
        name=f'{name}_',
    )
    model.graph.node.insert(0, node)
    model.graph.value_info.insert(0, value_info)

add_const_tiny(model, 4)
add_const_tiny(model, 3)
# add_const(model, 0)

model.graph.value_info.insert(-1, model.graph.output[0])
# print(model.graph.initializer)
print('============input:', model.graph.input)
# print(model.graph.value_info)

# print(model)
onnx.checker.check_model(model, full_check=True)
onnx.save(model, 'tiny_lstm.onnx')
#simplify('tiny_lstm.onnx', skipped_optimizers=['extract_constant_to_initializer'])
# print(model)


import onnxruntime as ort
import numpy as np
x = np.random.rand(2, 1, 4).astype(np.float32)
ort_sess = ort.InferenceSession('tiny_lstm.onnx')
outputs = ort_sess.run(None, {'input': x})

y = outputs
print(f'output: {y[0].shape}: {y}')

# assert False
##############

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
        dst = idx + 5 * i + 1

        old_output = model.graph.node[dst - 1].output[0]
        new_output = f'{old_output}_{dst}'
        model.graph.node[dst - 1].output[0] = new_output

        starts = f'starts_{dst}'
        ends = f'ends_{dst}'
        axes = f'axes_{dst}'
        steps = f'steps_{dst}'

        node = onnx.helper.make_node(
            'Slice',
            inputs=[new_output, starts, ends, axes, steps],
            outputs=[old_output],
        )

        model.graph.node.insert(dst, node)

        add_const(dst, [0], starts)
        add_const(dst, [2 ** 63 - 1], ends)
        add_const(dst, [2], steps)
        add_const(dst, [2], axes)


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
        dst = idx + 6 * i  # -1 + 7 = 6

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
        dst = idx + 2 * i
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

name = 'recognition'
simplify(name + ".onnx", input_shapes={'x': [1, 3, 32, 100]})

model = onnx.load(name + '.onnx')

for i, node in enumerate(model.graph.input):
    if node.name == 'Constant_0' or node.name == 'Constant_2':
        print(i, node.name)

# name        val  init input
# Constant_0: 6.0, #20  #21
# Constant_2: 0.0, #22  #23

del model.graph.initializer[22]
del model.graph.initializer[20]

del model.graph.input[23]
del model.graph.input[21]

C_0 = onnx.helper.make_tensor_value_info('Constant_0', onnx.TensorProto.FLOAT, [1])
c_0 = onnx.helper.make_node(
    'Constant', [], ['Constant_0'],
    value=onnx.numpy_helper.from_array(np.array([6], dtype=np.float32), name='Constant_0'),
)

C_1 = onnx.helper.make_tensor_value_info('Constant_2', onnx.TensorProto.FLOAT, [1])
c_1 = onnx.helper.make_node(
    'Constant', [], ['Constant_2'],
    value=onnx.numpy_helper.from_array(np.array([0], dtype=np.float32), name='Constant_2'),
)

model.graph.node.insert(0, c_0)
model.graph.node.insert(0, c_1)

model.graph.value_info.insert(0, C_0)
model.graph.value_info.insert(0, C_1)

fix_convs(model)
# del model.graph.output[0]
# model.graph.output.append(onnx.helper.make_tensor_value_info('save_infer_model/scale_0.tmp_1', onnx.TensorProto.FLOAT, [1, 2]))

del model.graph.output[0]
model.graph.output.append(
    onnx.helper.make_tensor_value_info('save_infer_model/scale_0.tmp_1', onnx.TensorProto.FLOAT, [1, 25, 6625]))


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

    for i, idx in enumerate(sorted(inputs)):  # hashset by deafult?
        dst = idx + i  # -1 + 2 = 1

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

        # for x in range(4):
        #     node.input[3 + x] = ''

    for i, idx in enumerate(broken):
        dst = idx + 9 * i  # 0 + 7 + 2

        node_f = model.graph.node[dst]

        assert node_f.attribute[0].name == 'direction'
        del node_f.attribute[0]
        # node_f.output[1] = ''
        # node_f.output[2] = ''
        node_f.attribute.append(onnx.helper.make_attribute('direction', 'forward'))

        node_b = copy.deepcopy(node_f)
        node_b.name += '_b'
        node_b.output[1] += '_b'
        node_b.output[2] += '_b'
        orig_h = node_f.input[5] # == orig_c
        print(orig_h)
        change_input(node_f, '_f')
        change_input(node_b, '_b')

        fh_id = -1
        bh_id = -1
        for id, node in enumerate(model.graph.initializer):
            if node.name == orig_h + '_f':
                print(f'=============={id}{node.name}')
                fh_id = id
            elif node.name == orig_h + '_b':
                bh_id = id
        for id in reversed(sorted([fh_id, bh_id])):
            if id != -1:
                print('Replacin=======================: ', model.graph.initializer[id].name)
                add_const_tiny(model, id)
                dst += 1

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
            inputs=[new_output_f, new_output_b],
            outputs=[old_output],
            axis=1
        )
        model.graph.node.insert(dst, node)

        model.graph.node.insert(dst, node_b)

        starts = f'starts_lstm_{dst}'
        ends = f'ends_lstm_{dst}'
        axes = f'axes_lstm_{dst}'
        steps = f'steps_lstm_{dst}'

        node = onnx.helper.make_node(
            'Slice',
            inputs=[old_input, starts, ends, axes, steps],
            outputs=[new_input],
        )

        model.graph.node.insert(dst, node)

        add_const(dst, [2 ** 31 - 1], starts)
        add_const(dst, [-2 ** 31 + 2], ends)
        add_const(dst, [-1], steps)
        add_const(dst, [0], axes)

        print(node_f, node_b)

def fix_squeeze(model):
    broken = []

    for i, node in enumerate(model.graph.node):
        if node.op_type == 'Squeeze' or node.op_type == 'Unsqueeze':
            broken.append(i)

    for i, idx in enumerate(broken):
        dst = idx #- i # -1
        node = model.graph.node[dst]
        out_name = node.output[0]
        valinfo = None
        for x in model.graph.value_info:
            if x.name == out_name:
                valinfo = x
        new_shape_ = [x.dim_value for x in valinfo.type.tensor_type.shape.dim]
        new_shape = onnx.numpy_helper.from_array(np.array(new_shape_, dtype = np.int64), name=out_name + '/new_shape')
        model.graph.initializer.insert(0, new_shape)
        new_node = onnx.helper.make_node(
            'Reshape',
            name = f'R{dst}',
            inputs=[node.input[0], out_name + '/new_shape'],
            outputs=[out_name],
        )
        del model.graph.node[dst]
        model.graph.node.insert(dst, new_node)
        print(f'{dst}, {out_name}: {new_shape_}, {node.input}')


def fix_softmax(model):
    # 1 25 96 -> 25 1 96

    new_shape = onnx.numpy_helper.from_array(np.array([25, 1, 96], dtype=np.int64), name='orig_shape')
    model.graph.initializer.insert(0, new_shape)

    new_node = onnx.helper.make_node(
        'Reshape',
        name=f'R00',
        inputs=['x0', 'orig_shape'],
        outputs=[model.graph.input[0].name],
    )
    model.graph.node.insert(0, new_node)

    new_node = onnx.helper.make_node(
        'Reshape',
        name=f'R01',
        inputs=['x1', 'orig_shape'],
        outputs=[model.graph.input[1].name],
    )
    model.graph.node.insert(0, new_node)

    print(model.graph.input)
    del model.graph.input[:]
    print(model.graph.input)
    model.graph.input.append(onnx.helper.make_tensor_value_info('x0', onnx.TensorProto.FLOAT, [1, 25, 96]))
    model.graph.input.append(onnx.helper.make_tensor_value_info('x1', onnx.TensorProto.FLOAT, [1, 25, 96]))

    broken = []
    for i, node in enumerate(model.graph.node):
        if node.op_type == 'Softmax':
            broken.append(i)

    for i, idx in enumerate(broken):
        dst = idx + 4 * i
        node = model.graph.node[dst]

        out_name = node.output[0]
        valinfo = None
        for x in model.graph.value_info:
            if x.name == node.input[0]:
                valinfo = x
        old_shape = [x.dim_value for x in valinfo.type.tensor_type.shape.dim]
        new_shape = copy.deepcopy(old_shape)
        new_shape.insert(0, 1)
        old_shape = onnx.numpy_helper.from_array(np.array(old_shape, dtype=np.int64), name=out_name + '/old_shape')
        new_shape = onnx.numpy_helper.from_array(np.array(new_shape, dtype=np.int64), name=out_name + '/new_shape')
        model.graph.initializer.insert(0, old_shape)
        model.graph.initializer.insert(0, new_shape)
        new_node = onnx.helper.make_node(
            'Reshape',
            name = f'R1{dst}',
            inputs=[node.input[0], out_name + '/new_shape'],
            outputs=[f'{node.input[0]}/reshape'],
        )
        model.graph.node.insert(dst, new_node)
        new_node = onnx.helper.make_node(
            'Transpose',
            name = f'T1{dst + 1}',
            inputs=[f'{node.input[0]}/reshape'],
            outputs=[f'{node.input[0]}/transpose'],
            perm=[0, 3, 1, 2]
        )
        model.graph.node.insert(dst + 1, new_node)



        # TODO: save output name

        new_node = onnx.helper.make_node(
            'Transpose',
            name = f'T2{dst + 3}',
            inputs=[f'{node.output[0]}/softmax'],
            outputs=[f'{node.output[0]}/transpose'],
            perm=[0, 3, 2, 1]
        )
        model.graph.node.insert(dst + 3, new_node)

        new_node = onnx.helper.make_node(
            'Reshape',
            name=f'R2{dst + 4}',
            inputs=[f'{node.output[0]}/transpose', out_name + '/old_shape'],
            outputs=[node.output[0]],
        )
        model.graph.node.insert(dst + 4, new_node)

        node.input[0] = f'{node.input[0]}/transpose'
        node.output[0] = f'{node.output[0]}/softmax'
        print(node.attribute)
        node.attribute[0].i = 1


        # reshape to 1, x, y, z
        # transpose 0 3 1 2
        # softmax axis = 1
        # transpose 0 2 3 1
        # reshape x, y, z

def workaround_matmul(model):
    i = -1
    for id, node in enumerate(model.graph.initializer):
        if node.name == 'ctc_fc_w_attr':
            i = id
            print(f'FOUND {i}')
    node = model.graph.initializer[i]
    del model.graph.initializer[i]
    data = onnx.numpy_helper.to_array(node)
    data = np.expand_dims(data, axis=0)

    inp_a = onnx.helper.make_node(
        'Constant',
        [],
        [f'add_a_{i}'],
        value=onnx.helper.make_tensor(
            name=f'add_a_{i}',
            data_type=onnx.TensorProto.FLOAT,
            dims=(1,),
            vals=np.array([0.]).astype(np.float32),
        )
    )

    inp_b = onnx.helper.make_node(
        'Constant',
        [],
        [f'add_b_{i}'],
        value=onnx.helper.make_tensor(
            name=f'add_b_{i}',
            data_type=onnx.TensorProto.FLOAT,
            dims=data.shape,
            vals=data.flatten().astype(np.float32)
        )
    )

    new_node = onnx.helper.make_node(
        'Add',
        [f'add_a_{i}', f'add_b_{i}'],
        [node.name]
    )

    idx = -1
    for id, n in enumerate(model.graph.node):
        if n.output[0] == 'ctc_fc.tmp_0':
            idx = id
            break
    model.graph.node.insert(idx, new_node)
    model.graph.node.insert(idx, inp_a)
    model.graph.node.insert(idx, inp_b)

    for id, n in enumerate(model.graph.initializer):
        if n.name == 'ctc_fc_w_attr':
        #if n.output[0] == 'ctc_fc_w_attr':
            print(f'found {i}')
            print(n)
    for n in model.graph.output:
        print(f'out: {n.name}')
    # model.graph.initializer.insert(0, onnx.numpy_helper.from_array(data, node.name))


fix_hardsigmoid(model)
fix_tranpose(model)
fix_lstm(model)

#fix_softmax(model)

inputs = model.graph.input
name_to_input = {}
for input in inputs:
    name_to_input[input.name] = input

for initializer in model.graph.initializer:
    if initializer.name in name_to_input:
        inputs.remove(name_to_input[initializer.name])


check_save(model, name + '_fixed.onnx')
model = onnx.load(name + '_fixed.onnx')
fix_squeeze(model)
#s = len(model.graph.value_info)
#for i in range(s):
#    del model.graph.value_info[-1]
check_save(model, name + '_fixed.onnx')
model = onnx.load(name + '_fixed.onnx')

def cut_model(model, from_names, to_name):
    new_inputs = []
    out_node = None
    for node in model.graph.node:
        if node.output[0] in from_names:
            new_inputs.append(node)
        elif node.output[0] == to_name:
            out_node = node



    visited = []
    new_nodes = []
    def toposort(node):
        if node in visited:
            return
        visited.append(node)
        for inp in node.input:
            for new_node in model.graph.node:
                if inp in new_node.output and new_node not in new_inputs:
                    toposort(new_node)
        new_nodes.append(node)
    toposort(out_node)
    print([x.output[0] for x in new_nodes])
    print(new_inputs)
    #print(model.graph.node[0].name, new_nodes[-1].name)

    #x = model.graph.input[0]
    for i in range(len(model.graph.input)):
        del model.graph.input[-1]
    #model.graph.input.append(x)
    for x in new_inputs:
        for v in model.graph.value_info:
            if v.name == x.output[0]:
                model.graph.input.append(v)
    #model.graph.input = [x.input[0] for x in new_inputs]

    # NOTE: comment fot 3rd part
    #for i in range(len(model.graph.output)):
    #    del model.graph.output[-1]
    #val_info = None
    #for x in model.graph.value_info:
    #    if x.name == out_node.output[0]:
    #        val_info = x
    #model.graph.output.append(val_info)

    print(f'input: {model.graph.input}, output: {model.graph.output}')
    #model.graph.output = [out_node.output[0]]
    for i in range(len(model.graph.node)):
        del model.graph.node[-1]
    for x in new_nodes:
        model.graph.node.append(x)
    #model.graph.node = new_nodes
#cut_model(model, from_names=['x'], to_name='transpose_1.tmp_0')
#check_save(model, name + 'fixed_part1.onnx')
#cut_model(model, from_names=['transpose_1.tmp_0', 'transpose_1.tmp_0_254'], to_name='Reshape_18')
#check_save(model, name + 'fixed_part2.onnx')
cut_model(model, from_names=['Reshape_18', 'Reshape_18_264'], to_name='save_infer_model/scale_0.tmp_1')

check_save(model, name + 'fixed_part3.onnx')
model = onnx.load(name + 'fixed_part3.onnx')
fix_softmax(model)
workaround_matmul(model)
check_save(model, name + 'fixed_part3.onnx')
# simplify(name + "_fixed.onnx", input_shapes={'x': [1, 3, 32, 100]})
import onnxruntime as ort
import numpy as np
x1 = np.random.rand(1, 25, 96).astype(np.float32)
ort_sess = ort.InferenceSession('recognitionfixed_part3.onnx')
outputs = ort_sess.run(None, {'x0': x1, 'x1' : x1})
print(outputs[0].shape)
print(model.graph.initializer[0])