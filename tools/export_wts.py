"""Export the supplied three-head RAICOM YOLOv5 WTS layout to a 640 x 640 ONNX model."""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def read_weights(path):
    tensors = {}
    with path.open(encoding="ascii") as stream:
        count = int(stream.readline())
        for line in stream:
            name, size, values = line.split(maxsplit=2)
            weights = np.frombuffer(bytes.fromhex(values), dtype=">f4").astype(np.float32)
            if name in tensors or weights.size != int(size):
                raise ValueError("Invalid WTS tensor {}".format(name))
            tensors[name] = weights
    if len(tensors) != count:
        raise ValueError("WTS tensor count does not match the header")
    return tensors


def make_model(weights):
    nodes, initializers = [], []

    def constant(value, dtype=np.float32):
        name = "weight_{}".format(len(initializers))
        initializers.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name=name))
        return name

    def node(kind, *inputs, **attributes):
        name = "feature_{}".format(len(nodes))
        nodes.append(helper.make_node(kind, list(inputs), [name], **attributes))
        return name

    def conv(image, prefix, kernel=1, stride=1):
        scale = weights[prefix + ".bn.weight"] / np.sqrt(weights[prefix + ".bn.running_var"] + 1e-3)
        bias = weights[prefix + ".bn.bias"] - weights[prefix + ".bn.running_mean"] * scale
        filters = weights[prefix + ".conv.weight"].reshape(len(scale), -1, kernel, kernel)
        folded = filters * scale[:, None, None, None]
        result = node("Conv", image, constant(folded), constant(bias),
                      pads=[(kernel - 1) // 2] * 4, strides=[stride] * 2)
        return node("Mul", result, node("Sigmoid", result))

    def c3(image, prefix, shortcut):
        branch = conv(image, prefix + ".cv1")
        bypass = conv(image, prefix + ".cv2")
        index = 0
        while "{}.m.{}.cv1.conv.weight".format(prefix, index) in weights:
            name = "{}.m.{}".format(prefix, index)
            result = conv(conv(branch, name + ".cv1"), name + ".cv2", kernel=3)
            branch = node("Add", branch, result) if shortcut else result
            index += 1
        if index == 0:
            raise ValueError("Missing YOLOv5 bottlenecks in {}".format(prefix))
        return conv(node("Concat", branch, bypass, axis=1), prefix + ".cv3")

    def sppf(image):
        maps = [conv(image, "model.9.cv1")]
        for _ in range(3):
            maps.append(node("MaxPool", maps[-1], kernel_shape=[5, 5], pads=[2, 2, 2, 2]))
        return conv(node("Concat", *maps, axis=1), "model.9.cv2")

    def upsample(image):
        return node("Resize", image, constant([]), constant([1, 1, 2, 2]), mode="nearest",
                    coordinate_transformation_mode="asymmetric", nearest_mode="floor")

    # Backbone and feature pyramid from the supplied TensorRTx YOLOv5 builder.
    feature = conv("images", "model.0", kernel=6, stride=2)
    feature = c3(conv(feature, "model.1", kernel=3, stride=2), "model.2", True)
    low = c3(conv(feature, "model.3", kernel=3, stride=2), "model.4", True)
    middle = c3(conv(low, "model.5", kernel=3, stride=2), "model.6", True)
    feature = c3(conv(middle, "model.7", kernel=3, stride=2), "model.8", True)
    high = conv(sppf(feature), "model.10")
    feature = c3(node("Concat", upsample(high), middle, axis=1), "model.13", False)
    bridge = conv(feature, "model.14")
    small = c3(node("Concat", upsample(bridge), low, axis=1), "model.17", False)
    feature = conv(small, "model.18", kernel=3, stride=2)
    medium = c3(node("Concat", feature, bridge, axis=1), "model.20", False)
    feature = conv(medium, "model.21", kernel=3, stride=2)
    large = c3(node("Concat", feature, high, axis=1), "model.23", False)

    def channels(tensor, start, end):
        return node("Slice", tensor, constant([start], np.int64), constant([end], np.int64),
                    constant([4], np.int64))

    anchors = weights["model.24.anchor_grid"].reshape(3, 3, 2)
    heads = []
    for index, (features, stride) in enumerate(zip((small, medium, large), (8, 16, 32))):
        prefix = "model.24.m.{}".format(index)
        fields, size = 14, 640 // stride
        bias = weights[prefix + ".bias"]
        if bias.size != 3 * fields:
            raise ValueError("Expected the three-anchor, nine-class RAICOM detection heads")
        filters = weights[prefix + ".weight"].reshape(bias.size, -1, 1, 1)
        logits = node("Conv", features, constant(filters), constant(bias))
        grid_shape = constant([1, 3, fields, size, size], np.int64)
        activation = node("Sigmoid", node("Transpose", node("Reshape", logits, grid_shape),
                                          perm=[0, 1, 3, 4, 2]))
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        grid = constant(np.stack((x, y), axis=-1)[None, None] - 0.5)
        xy = node("Mul", node("Add", node("Mul", channels(activation, 0, 2), constant(2)), grid),
                  constant(stride))
        lengths = node("Mul", channels(activation, 2, 4), constant(2))
        wh = node("Mul", node("Mul", lengths, lengths), constant(anchors[index].reshape(1, 3, 1, 1, 2)))
        prediction = node("Concat", xy, wh, channels(activation, 4, fields), axis=4)
        heads.append(node("Reshape", prediction, constant([1, -1, fields], np.int64)))

    result = node("Concat", *heads, axis=1)
    graph = helper.make_graph(nodes, "RAICOM YOLOv5", [helper.make_tensor_value_info(
        "images", TensorProto.FLOAT, [1, 3, 640, 640])], [helper.make_tensor_value_info(
        result, TensorProto.FLOAT, [1, 25200, 14])], initializers)
    model = helper.make_model(graph, ir_version=9, opset_imports=[helper.make_opsetid("", 12)])
    onnx.checker.check_model(model, full_check=True)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        onnx.save_model(make_model(read_weights(args.weights)), str(args.output))
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, "Error: {}\n".format(error))
    print(args.output)
