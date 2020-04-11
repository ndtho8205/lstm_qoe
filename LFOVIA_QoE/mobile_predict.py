from math import sqrt

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from sklearn.metrics import mean_squared_error
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr


input_data = np.loadtxt(f"testX_1_134_1_3.txt").reshape((134, 1, 3))
actual_data = np.loadtxt("./testY_1.txt")
GRAPH_PB_PATH = "./mobile_model.pb"
with tf.Session() as sess:
    print("load graph")
    with gfile.FastGFile(GRAPH_PB_PATH, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name="")
        graph_nodes = [n for n in graph_def.node]
        names = []
        for t in graph_nodes:
            print(t.name, t.input)
    output_tensor = sess.graph.get_tensor_by_name("time_distributed_2/transpose_1:0")

    predictions = []
    for i in range(134):
        predictions.append(sess.run(output_tensor, {"lstm_3_input:0": [input_data[i]]}))

    predictions = np.array(predictions).flatten()
    print(pearsonr(predictions, actual_data))
    print(spearmanr(predictions, actual_data))
    print(sqrt(mean_squared_error(predictions, actual_data)))

    from matplotlib import pyplot as plt

    plt.plot(predictions, label="Predicted QoE on mobile", color="red")
    plt.plot(actual_data, label="Actual QoE", color="blue")
    plt.legend()
    plt.show()
    print(predictions)
