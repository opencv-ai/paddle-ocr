import onnx
import onnxsim
import numpy as np
from onnx import numpy_helper, helper, AttributeProto, TensorProto, GraphProto

def simplify(name, rename=False, **kwargs):
    model, check = onnxsim.simplify(name, **kwargs)
    assert check, "couldn't valide"
    name = name[:-5]
    if rename:
        name += '_optimized'
    onnx.save(model, name + '.onnx')
    return model

def check_save(model, name):
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, name)


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


def fix_resize(model):
    # ToDo: Rename outputs
    for i, node in enumerate(model.graph.node):
        if node.op_type == "Resize":
            print(f'fixing Resize "{node.name}"')
            to_delete = -1
            for j, attribute in enumerate(node.attribute):
                if attribute.name == "mode":
                    attribute.s = b"linear"
                if attribute.name == "nearest_mode":
                    to_delete = j
                if attribute.name == "coordinate_transformation_mode":
                    attribute.s = b"align_corners"
            if to_delete != -1:
                del node.attribute[to_delete]


if __name__ == "__main__":
    model_path = "./source/en_PP-OCRv3_det_infer_fixed_shape.onnx"
    model = simplify(model_path, True)
    # model = onnx.load("../../detection_v3_optimized.onnx")
    inputs_to_delete = []
    initializers_to_delete = []
    for i, node in enumerate(model.graph.input):
        if node.name == 'Constant_0' or node.name == 'Constant_2':
            print(f"Input: {i} {node.name}")
            inputs_to_delete.append(i)
    for i, node in enumerate(model.graph.initializer):
        if node.name == 'Constant_0' or node.name == 'Constant_2':
            print(f"Initializer: {i} {node.name}")
            initializers_to_delete.append(i)
        # if node.name == 'Constant_0' or node.name == 'Constant_2':
    # fix_clip(model)
    for i in sorted(inputs_to_delete, reverse=True):
        del model.graph.input[i]
    for i in sorted(initializers_to_delete, reverse=True):
        del model.graph.initializer[i]
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
    # onnx.save(model, "../../detection_v3_fixed_final.onnx")
    # fix_convs(model)
    # for i, node in enumerate(model.graph.value_info):
    #     print(i, node.name, node.type.tensor_type.shape)

    # model, check = onnxsim.simplify(model)
    fix_hardsigmoid(model)
    # fix_resize(model)
    check_save(model, "./changed/en_PP-OCRv3_det_infer_fixed_shape_optimized_fixed.onnx")
    # fix_hardsigmoid(model)