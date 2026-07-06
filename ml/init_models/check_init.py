import onnxruntime as ort

session = ort.InferenceSession("init.onnx")

print("Inputs:")
for x in session.get_inputs():
    print(x.name, x.shape, x.type)

print()

print("Outputs:")
for x in session.get_outputs():
    print(x.name, x.shape, x.type)