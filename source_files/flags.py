import tensorflow as tf
from pathlib import Path

# Common flags

# Inputs params
tf.flags.DEFINE_integer('class_nbr', 6,
                        lower_bound=0, upper_bound=6,
                        help='DAGM class to train on.')
tf.flags.DEFINE_enum('mode', 'cached', ['direct', 'cached'],
                     help='''The input pipeline mode, either direct ent to end '''
                          '''forward pass, or through cached CNN output.''')
tf.flags.DEFINE_integer('target_width', 224,
                        lower_bound=0,
                        help='End width after transforming images.')
tf.flags.DEFINE_string('data_dir', str(Path(__file__).parent / '../data/DAGM 2007 - Splitted'),
                       help='Where the images are stored (by class and by train / test).')
# Model params
tf.flags.DEFINE_enum('kernel', 'linear', ['linear', 'rbf', 'rffm'],
                     help='Approximation of the SVM kernel.')
tf.flags.DEFINE_float('learning_rate', 0.1,
                      lower_bound=0,
                      help='Starting value for the learning rate.')
tf.flags.DEFINE_float('c', 3,
                      lower_bound=0,
                      help='C parameter for the hardness of the margin.')
tf.flags.DEFINE_enum('type', 'svdd', ['svdd', 'ocsvm'],
                     help='Type of the classifier: SVDD (hypersphere) or OCSVM (hyperplane).')

# Train / test params
tf.flags.DEFINE_string('cnn_output_dir', str(Path(__file__).parent / '../tmp/cnn_output/VGG16'),
                       help='Where the cached files are located, if using the cached mode')
tf.flags.DEFINE_integer('batch_size', 64,
                        lower_bound=1,
                        help='Batch size.')
tf.flags.DEFINE_integer('epochs', 100,
                        lower_bound=0,
                        help='Number of epochs.')
tf.flags.DEFINE_string('model_dir', str(Path(__file__).parent / '../tmp/estimator'),
                       help='Where to write events and checkpoints.')


FLAGS = tf.flags.FLAGS
