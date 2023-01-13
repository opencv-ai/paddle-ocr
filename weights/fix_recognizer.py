import onnx
import onnxsim
import numpy as np
from onnx import numpy_helper, helper, AttributeProto, TensorProto, GraphProto
import onnxruntime as ort

# model, check = onnxsim.simplify('./source/en_PP-OCRv3_rec_infer_fixed_shape.onnx', input_shapes={'x': [1, 3, 48, 1216]})
# # assert check, "couldn't valide"
# onnx.save(model, './source/en_PP-OCRv3_rec_infer_fixed_shape_simplified.onnx')


model = onnx.load('./source/en_PP-OCRv3_rec_infer_fixed_shape_simplified.onnx')

for i, node in enumerate(model.graph.input):
    if node.name == 'Constant_0' or node.name == 'Constant_2':
        print('inp: ', i, node.name)

for i, node in enumerate(model.graph.initializer):
    if node.name == 'Constant_0' or node.name == 'Constant_2':
        print('init: ', i, node.name)        


# name        val  init input
#inp:  9 Constant_0
#inp:  11 Constant_2
#init:  8 Constant_0
#init:  10 Constant_2
# Constant_0: 6.0, #20  #21
# Constant_2: 0.0, #22  #23
del model.graph.initializer[24]
del model.graph.initializer[22]
#
# del model.graph.input[11]
# del model.graph.input[9]


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
print('\n\n')
for i, node in enumerate(model.graph.initializer):
    if node.name == "Constant_123" or node.name == "Constant_124" or node.name == "Constant_130" or node.name == "Constant_134":
        print('init: ', i, node.name)

del model.graph.initializer[31]
del model.graph.initializer[30]
del model.graph.initializer[27]
del model.graph.initializer[26]

C_123 = onnx.helper.make_tensor_value_info('Constant_123', onnx.TensorProto.INT64, [1])
c_123 = onnx.helper.make_node(
    'Constant', [], ['Constant_123'],
    value=onnx.numpy_helper.from_array(np.array([0], dtype=np.int64), name='Constant_123'),
)

C_124 = onnx.helper.make_tensor_value_info('Constant_124', onnx.TensorProto.INT64, [1])
c_124 = onnx.helper.make_node(
    'Constant', [], ['Constant_124'],
    value=onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name='Constant_124'),
)

C_130 = onnx.helper.make_tensor_value_info('Constant_130', onnx.TensorProto.INT64, [1])
c_130 = onnx.helper.make_node(
    'Constant', [], ['Constant_130'],
    value=onnx.numpy_helper.from_array(np.array([2], dtype=np.int64), name='Constant_130'),
)

C_134 = onnx.helper.make_tensor_value_info('Constant_134', onnx.TensorProto.INT64, [1])
c_134 = onnx.helper.make_node(
    'Constant', [], ['Constant_134'],
    value=onnx.numpy_helper.from_array(np.array([3], dtype=np.int64), name='Constant_134'),
)

model.graph.node.insert(0, c_123)
model.graph.node.insert(0, c_124)

model.graph.value_info.insert(0, C_123)
model.graph.value_info.insert(0, C_124)

model.graph.node.insert(0, c_130)
model.graph.node.insert(0, c_134)

model.graph.value_info.insert(0, C_130)
model.graph.value_info.insert(0, C_134)

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
                    if at.ints[0] < at.ints[1]:
                        at.ints[1] = at.ints[0]
                        broken.append((i, True))
                    else:
                        at.ints[0] = at.ints[1]
                        broken.append((i, False))

    for i, (idx, swap_axes) in enumerate(broken):
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
        add_const(dst, [2 if not swap_axes else 3], axes)

        
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
            if len(node.attribute[0].ints) < 4:
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


def fix_5_dim_nodes(model):
    for i, node in enumerate(model.graph.node):
        if node.op_type == 'Reshape':
            for val in model.graph.initializer:
                if val.name == node.input[1]:
                    if val.dims[0] == 5:
                        del val.int64_data[0]
                        val.dims[0] = 4
                        # val.int64_data[0] = -1 # ToDo: produces shape mismatch error without this line
    for i, node in enumerate(model.graph.node):
        if node.op_type == "Transpose" and len(node.attribute[0].ints) == 5:
            del node.attribute[0].ints[0]
            node.attribute[0].ints[0] = 1
            node.attribute[0].ints[1] = 2
            node.attribute[0].ints[2] = 0
            node.attribute[0].ints[3] = 3

def fix_lstm(model):
    constify = set()
    lstms = []
    for i, node in enumerate(model.graph.node):
        if node.op_type == 'LSTM':
            lstms.append(i)
            tmp = set(node.input[-2:])
            for i, node2 in enumerate(model.graph.initializer):
                if node2.name in tmp:
                    constify.add(i)
    constify = list(sorted(constify))
    for idx in reversed(constify):
        name = model.graph.initializer[idx].name
        data = onnx.numpy_helper.to_array(model.graph.initializer[idx])
        del model.graph.initializer[idx]
        add_const(0, data, name, data.dtype)

def fix_squeeze(model):
    node_to_delete = []
    for i, node in enumerate(model.graph.node):
        if node.op_type == "Squeeze":
            input_name = node.input[0]
            valinfo = None
            for x in model.graph.value_info:
                if x.name == input_name:
                    valinfo = x
            if len(valinfo.type.tensor_type.shape.dim) == 5:
                node_to_delete.append(i)
                for j, node2 in enumerate(model.graph.node):
                    if node2.output[0] in node.input:
                        model.graph.node[j].output[0] = node.output[0]
    for i in node_to_delete[::-1]:
        del model.graph.node[i]

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
        

def fix_final_softmax(model):
    # reshape to y, z
    # softmax axis = 1
    # reshape 1, y, z
    
    idx = -1
    for i, node in enumerate(model.graph.node):
        if node.op_type == 'Softmax':
            idx = i
    
    orig_in_name = model.graph.node[idx].input[0]
    orig_out_name = model.graph.node[idx].output[0]

    old_shape = [x.dim_value for x in model.graph.output[0].type.tensor_type.shape.dim]
    old_axis = model.graph.node[idx].attribute[0].i
    new_axis = 1
    new_shape = onnx.numpy_helper.from_array(np.array(old_shape, dtype=np.int64)[1:], name=orig_out_name + '/new_shape')
    old_shape = onnx.numpy_helper.from_array(np.array(old_shape, dtype=np.int64), name=orig_out_name + '/old_shape')
    
    model.graph.initializer.insert(0, new_shape)
    model.graph.initializer.insert(0, old_shape)
    
    jdx = -1
    for i, node in enumerate(model.graph.node):
        if orig_in_name in node.output:
            node.output[0] = orig_in_name + '/old'
            jdx = i
    
    # Squeeze input
    new_node = onnx.helper.make_node(
        'Reshape',
        name=f'R00',
        inputs=[orig_in_name + '/old', orig_out_name + '/new_shape'],
        outputs=[orig_in_name],
    )
    model.graph.node.insert(idx, new_node)
    idx += 1
    
    # Set axis to 1
    model.graph.node[idx].attribute[0].i = new_axis
    model.graph.node[idx].output[0] = orig_out_name + '/old'
    
    # Squeeze output
    new_node = onnx.helper.make_node(
        'Reshape',
        name=f'R01',
        inputs=[orig_out_name + '/old', orig_out_name + '/old_shape'],
        outputs=[orig_out_name],
    )
    model.graph.node.insert(idx + 1, new_node)

    
def fix_reshape(model):
    broken = []
    for i, node in enumerate(model.graph.node):
        if node.op_type == 'Reshape':
            for val in model.graph.initializer:
                if val.name == node.input[1]:
                    print(val.name)
                    data = onnx.numpy_helper.to_array(val)
                    print(data)
                    if 0 in data:
                        for info in model.graph.value_info:
                            if info.name == node.output[0]:
                                shape = [x.dim_value for x in info.type.tensor_type.shape.dim]
                                print(shape)
                                for j in range(len(shape)):
                                    val.int64_data[j] = shape[j]


def fix_avg_pool(model):
    for i, node in enumerate(model.graph.node):
        if node.op_type == "AveragePool":
            for j, att in enumerate(model.graph.node[i].attribute):
                if att.name == "count_include_pad":
                    del model.graph.node[i].attribute[j]
                    break

def fix_reduce_mean(model):
    processed = []
    # ToDo: Make correct fix with transpose nodes, this one is for Lens Studio importing only
    for i, node in enumerate(model.graph.node):
        if node.name in processed:
            continue
        dst = i
        if node.op_type == "ReduceMean":
            processed.append(node.name)
            for j, att in enumerate(model.graph.node[i].attribute):
                if att.name == "axes":
                    orig_out_name = node.output[0]
                    orig_input_name = node.input[0]
                    model.graph.node[i].output[0] = orig_out_name + "_permuted"
                    transpose_output_name = orig_input_name + "_permuted"
                    model.graph.node[i].input[0] = transpose_output_name
                    swap_axes = [0, 1, 2]
                    assert len(att.ints) == 1
                    swap_axes[1] = att.ints[0]
                    swap_axes[att.ints[0]] = 1
                    model.graph.node[i].attribute[j].ints[0] = 1
                    new_node = onnx.helper.make_node(
                        "Transpose",
                        name=f'ReduceMeanTranspose_{dst}',
                        inputs=[orig_input_name],
                        outputs=[transpose_output_name],
                        perm=swap_axes,
                    )
                    model.graph.node.insert(dst, new_node)
                    dst += 1
                    reverse_node = onnx.helper.make_node(
                        "Transpose",
                        name=f'ReduceMeanTransposeReverse_{dst}',
                        inputs=[orig_out_name + "_permuted"],
                        outputs=[orig_out_name],
                        perm=swap_axes,

                    )
                    model.graph.node.insert(dst + 1, reverse_node)

# def fix_matmul(model):
#     matmul = 'MatMul_0'
#     add = 'Add_31'
    
#     broken = []
#     for i, node in enumerate(model.graph.node):
#         if node.name in set([matmul, add]):
#             broken.append(i)
#     for i in reversed(broken):
#         del model.graph.node[i]
    
#     new_node = onnx.helper.make_node(
#         'Gemm',
#         name=f'Gemm',
#         inputs=['transpose_5.tmp_0', 'linear_2.w_0', 'linear_2.b_0'],
#         outputs=['linear_6.tmp_1'],
#         alpha=1.,
#         beta=1.,
#         transA=0,
#         transB=0
#     )
    

#     for i, node in enumerate(model.graph.node):
#         if node.output[0] == 'linear_6.tmp_1':
#             del model.graph.node[i]
    
#     model.graph.node.insert(min(broken), new_node)
#     idx = -1
#     for i, val in enumerate(model.graph.initializer):
#         if val.name == 'transpose_5.tmp_0/new_shape':
#             idx = i
#             break
    
#     data = onnx.numpy_helper.to_array(model.graph.initializer[idx])
#     array = onnx.numpy_helper.from_array(data[1:], name=model.graph.initializer[idx].name)
    
#     print('HERE', model.graph.initializer[idx])
#     del model.graph.initializer[idx]

#     model.graph.initializer.insert(idx, array)
#     print('HERE', model.graph.initializer[idx])

def fix_matmul(model):
    names_to_fix = []
    processed = []
    for i, node in enumerate(model.graph.node):
        if node.name in processed:
            continue
        dst = i
        broken = False
        if node.op_type == 'MatMul':
            processed.append(node.name)
            for j, input_name in enumerate(node.input):
                valinfo = None
                valinfo_output = None
                for x in model.graph.value_info:
                    if x.name == input_name:
                        valinfo = x
                    if x.name == node.output[0]:
                        valinfo_output = x
                if valinfo is None or valinfo_output is None:
                    break
                if len(valinfo.type.tensor_type.shape.dim) == 4:
                    print("fixing ", node.name)
                    broken = True
                    new_shape_ = [k.dim_value for k in valinfo.type.tensor_type.shape.dim if k.dim_value != 1]
                    # new_shape_.insert(1, 0)
                    # new_shape = onnx.numpy_helper.from_array(np.array(new_shape_, dtype=np.int64),
                    #                                          name=input_name + '/new_shape')
                    # model.graph.initializer.insert(0, new_shape)
                    out_name = input_name + "_reshaped"
                    new_node = onnx.helper.make_node(
                        "Transpose",
                        name=f'MatMul_reshape{dst}',
                        inputs=[input_name],
                        outputs=[out_name],
                        perm=(1, 0, 2, 3),
                    )
                    # new_node = onnx.helper.make_node(
                    #     'Reshape',
                    #     name=f'MatMul_reshape{dst}',
                    #     inputs=[input_name, input_name + '/new_shape'],
                    #     outputs=[out_name],
                    # )
                    model.graph.node.insert(dst, new_node)
                    dst += 1
                    model.graph.node[dst].input[j] = out_name
            if broken:
                valinfo_output = None
                matmul_output_name = node.output[0]
                for x in model.graph.value_info:
                    if x.name == matmul_output_name:
                        valinfo_output = x
                matmul_after_reshape_shape_ = [k.dim_value for k in valinfo_output.type.tensor_type.shape.dim]
                matmul_after_reshape_shape = onnx.numpy_helper.from_array(
                    np.array(matmul_after_reshape_shape_, dtype=np.int64),
                    name=matmul_output_name + '/new_shape')
                out_name = matmul_output_name + "_reshaped"
                new_node = onnx.helper.make_node(
                    "Transpose",
                    name=f'MatMul_after_reshape{dst}',
                    inputs=[matmul_output_name],
                    outputs=[out_name],
                    perm=(1, 0, 2, 3),
                )
                # new_node = onnx.helper.make_node(
                #     'Reshape',
                #     name=f'MatMul_after_reshape{dst}',
                #     inputs=[matmul_output_name, matmul_output_name + '/new_shape'],
                #     outputs=[out_name],
                # )
                # model.graph.initializer.insert(0, matmul_after_reshape_shape)
                dst += 1
                model.graph.node.insert(dst, new_node)
                names_to_fix.append(matmul_output_name)
    for i, node in enumerate(model.graph.node):
        if node.op_type == "Transpose" and "MatMul" in node.name:
            continue
        for j, inp in enumerate(node.input):
            if inp in names_to_fix:
                node.input[j] = node.input[j] + "_reshaped"


def fix_softmax(model):
    # ToDo: apply correct fix, add Transpose nodes before and after softmax
    for i, node in enumerate(model.graph.node):
        if node.op_type == "Softmax":
            for j, att in enumerate(model.graph.node[i].attribute):
                if att.name == "axis" and att.i == 3:
                    print(node.name)
                    att.i = 1
fix_reduce_mean(model)
fix_hardsigmoid(model)
fix_convs(model)
fix_tranpose(model)
# fix_lstm(model)

model.graph.output[0].type.tensor_type.shape.dim[0].dim_value = 1

from onnx import shape_inference
model = shape_inference.infer_shapes(model)

fix_final_softmax(model)
for i in range(len(model.graph.value_info)):
    del model.graph.value_info[-1]
model = shape_inference.infer_shapes(model)
fix_squeeze(model)
fix_reshape(model)
fix_avg_pool(model)
fix_5_dim_nodes(model)
fix_matmul(model)
fix_softmax(model)



for i in range(len(model.graph.value_info)):
    del model.graph.value_info[-1]
model = shape_inference.infer_shapes(model)

for x in range(len(model.graph.input) - 1):
    del model.graph.input[-1]

all_inputs = set()
for node in model.graph.node:
    for input_name in node.input:
        all_inputs.add(input_name)

removal = []
for i, node in enumerate(model.graph.initializer):
    if node.name not in all_inputs:
        removal.append(i)
for i in reversed(removal):
    del model.graph.initializer[i]

model = shape_inference.infer_shapes(model)
onnx.checker.check_model(model, True)
dummy_input = np.random.random((1, 3, 48, 1216)).astype(np.float32)
onnx.save(model, './changed/recognition_v3.onnx')
ort_sess = ort.InferenceSession('./changed/recognition_v3.onnx')
outputs = ort_sess.run(None, {'x': dummy_input})
# model, check = onnxsim.simplify(model, input_shapes={'x': [1, 3, 32, 1216]})
print('ok')