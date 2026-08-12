import tensorflow as tf

logits = tf.random.normal((1, 6))
probs = tf.nn.softmax(logits)

print("Logits:     ", logits.numpy()[0])
print("Softmax:    ", probs.numpy()[0])
print("Argmax logits:", tf.argmax(logits, axis=-1).numpy()[0])
print("Argmax probs: ", tf.argmax(probs, axis=-1).numpy()[0])
print("Identical?", tf.argmax(logits, axis=-1).numpy()[0] == tf.argmax(probs, axis=-1).numpy()[0])