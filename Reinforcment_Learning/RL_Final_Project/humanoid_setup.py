import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model

class PolicyNetwork(Model):
    def __init__(self, obs_space_dim: int, action_space_dim: int):
        super().__init__()
        
        input_layer = keras.layers.Input((obs_space_dim,))
        hidden_layer1 = keras.layers.Dense(obs_space_dim * 3, activation="elu")(input_layer)
        hidden_layer2 = keras.layers.Dense(obs_space_dim * 2, activation="elu")(hidden_layer1)
        
        self.mean_output_layer = keras.layers.Dense(action_space_dim, activation="tanh")(hidden_layer2)
        self.std_output_layer = keras.layers.Dense(action_space_dim, activation="softmax")(hidden_layer2)
        
        self.model = Model(inputs=[input_layer], outputs=[self.mean_output_layer, self.std_output_layer])
        
    def call(self, state):
        return self.model(state)
    
    