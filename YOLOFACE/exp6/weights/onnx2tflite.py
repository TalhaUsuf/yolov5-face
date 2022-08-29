# import onnx
# from onnx_tf.backend import prepare
#
#
# onnx_model = onnx.load("YOLOFACE/exp6/weights/best_simplified.onnx")
# tf_rep = prepare(onnx_model)
# tf_model_path = "YOLOFACE/exp6/weights/best_simplified_TF"
# tf_rep.export_graph(tf_model_path)

# tf to tflite
import tensorflow as tf
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("/home/talha/oneTB/yolov5-face/YOLOFACE/exp6/weights/best_simplified.pb")
# converter optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter target device
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
# convert to fp16
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

# Save the tflite model
with open("best_simplified_fp16.tflite", "wb") as f:
    f.write(tflite_model)


# or use the onnx-tf tool to convert onnx to tflite