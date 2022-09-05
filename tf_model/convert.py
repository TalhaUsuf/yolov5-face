import tensorflow as tf
# import sclblonnx as so
from rich.console import Console
#
# ONNX_MODEL = "/home/talha/oneTB/yolov5-face/YOLOFACE/exp6/weights/best_simplified.onnx"
# g = so.graph_from_file(ONNX_MODEL)
# # so.display(g)
#
# so.check(g)
# Console().print(f"checks passed 🔷")
# g = so.clean(g)



# define tensorlfow model

input = tf.keras.Input(shape=(320, 320, 3) , name="input")
x_o = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2,2) ,name="Conv_0", padding='same')(input)
x_a = tf.keras.activations.sigmoid(x_o)
x_mul = tf.keras.layers.Multiply(name="Multiply_2")([x_o, x_a])

# branch after Mul_2
x = tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 1), strides=(1,1), padding='valid')(x_mul)
x_ = tf.keras.activations.sigmoid(x)
x = tf.keras.layers.Multiply()([x, x_])

x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2,2), padding='same')(x)
a = tf.keras.activations.sigmoid(x)
x = tf.keras.layers.Multiply()([x, a])


# for concat at Concat_10
b = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='valid')(x_mul)

# x -> [N, 80, 80, 16] , b -> [N, 80, 80, 16]
x = tf.keras.layers.Concatenate(axis=-1)([x, b]) # [N, 80, 80, 32]

# after Concat_10
xx = tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1,1), padding='valid')(x)



cc = tf.keras.activations.sigmoid(xx)
out = tf.keras.layers.Multiply()([xx, cc])

# Mul_13 begins from here


model = tf.keras.Model(inputs=[input], outputs=[x])

Console().print(f"model ⚡ build successfully 🔷")
Console().print(model.summary())