import os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Redirect stderr to devnull
import sys
import io
sys.stderr = io.StringIO()

import tensorflow as tf
import logging
import warnings

# Suppress absolutely everything
logging.getLogger().setLevel(logging.ERROR)
warnings.filterwarnings('ignore')
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Disable all TF logging
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Disable TF2 behavior
tf.compat.v1.disable_eager_execution()

# Disable deprecation warnings
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

# Suppress specific TF messages
import contextlib
import sys

# Create a context manager to suppress stdout/stderr
@contextlib.contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stderr(fnull) as err, contextlib.redirect_stdout(fnull) as out:
            yield (err, out)

# Use this context manager when loading and running the model
with suppress_stdout_stderr():

    import numpy as np

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def load_labels(label_file):
    labels = []
    with open(label_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def classify_image(image_path):
    # Update paths to be absolute or relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(script_dir, "retrained_graph.pb")
    label_file = os.path.join(script_dir, "retrained_labels.txt")
    
    input_layer = "Placeholder"
    output_layer = "final_result"
    
    # Load the model and labels
    graph = load_graph(model_file)
    labels = load_labels(label_file)
    
    # Debug: Print operations in the graph    
    # Update these to match the actual operations in your graph
    possible_input_names = [
        "import/DecodeJpeg/contents",  # This appears to be the input operation
        "import/input",
        "input",
        "Placeholder",
        "input_tensor"
    ]
    
    possible_output_names = [
        "import/final_result",
        "final_result",
        "import/Softmax",
        "Softmax"
    ]
    
    # Find the first matching input operation
    for name in possible_input_names:
        try:
            input_operation = graph.get_operation_by_name(name)
            input_name = name
            break
        except KeyError:
            continue
    else:
        raise ValueError("Could not find input operation in graph")
    
    # Find the first matching output operation
    for name in possible_output_names:
        try:
            output_operation = graph.get_operation_by_name(name)
            output_name = name
            break
        except KeyError:
            continue
    else:
        raise ValueError("Could not find output operation in graph")
    
    # Read the image file directly as bytes instead of preprocessing
    with tf.io.gfile.GFile(image_path, 'rb') as fid:
        image_data = fid.read()
    
    # Run the classification with raw image data
    with tf.compat.v1.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: image_data
        })
    
    # Get top results
    results = np.squeeze(results)
    top_index = results.argmax()  # Get index of highest confidence prediction
    
    # Return only the highest confidence prediction
    return {
        'label': labels[top_index],
        'confidence': float(results[top_index])
    }

# Example usage
if __name__ == "__main__":
    # Ask user for image path
    image_path = input("Enter input image path: ").strip('"')  # Strip quotes if present
    
    with suppress_stdout_stderr():
        prediction = classify_image(image_path)
    
    # Print only the highest confidence prediction
    print(f"Prediction: {prediction['label']} ({prediction['confidence']*100:.2f}%)")