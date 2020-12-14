import tensorflow as tf

category_indices = [0, 1, 2, 2, 1, 0]
unique_category_count = 3
inputs = tf.one_hot(category_indices, unique_category_count)
print(inputs)
with tf.Session() as sess:  print(inputs.eval())