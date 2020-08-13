from tensorflow import keras
from tensorflow.keras import backend as K

"""
--- CUSTOM MODEL LAYERS FOR KERAS & TENSORFLOW 2.0 ---

-- LayerNormLSTMCell --
LSTM Cell with layer normalization applied to features before gate activations

-- LayerNormRNNCell --
RNN Cell with layer normalization applied to features before output activation

"""

class LayerNormLSTMCell(keras.layers.Layer):
    """ LSTM Cell with Layer Normalization, built from weights.
    Apply as a layer using tensorflow.keras.layers.RNN(LayerNormLSTMCell(), return_sequence=True...)
    : param units: number units in cell to pass as number of hidden state units, c state units, and output units
    : param layer_normalization: boolean flag determining whether or not to perform layer normalization on the features
    : param layer_norm_c_state: boolean flag determining whether or not to perform layer normalization on the long term
        state (note this is overridden by layer_normalization=False)
    : param dropout_rate: float in [0,1) determining dropout level for cell inputs
    : param recurrent_dropout_rate: float in [0,1) determining dropout level for cell hidden states
    : param dropout_seed: random seed to ensure determinism
    """
    def __init__(self, units, layer_normalization:bool=True, layer_norm_c_state:bool=True,
                 dropout_rate:float=0.0, recurrent_dropout_rate:float=0.0, **kwargs):
        self.units = units
        self.state_size = [units, units] # hidden state h and cell state c
        print('state_size',self.state_size, 'units',self.units)#, 'batch_size',self.batch_size)
        assert 0 <= dropout_rate < 1, "dropout_rate required to be in [0,1)"
        self.dropout_rate = 0 #dropout_rate #TODO think this is applied in the wrong place - massively slowing training
        assert 0 <= recurrent_dropout_rate < 1, "recurrent_dropout_rate required to be in [0,1)"
        self.recurrent_dropout_rate = recurrent_dropout_rate
        self.layer_normalization = layer_normalization # Flag for whether or not to apply layer normalization
        self.layer_norm_c_state = layer_norm_c_state # Flag for whether or not to apply layer normalization to long term state
        self.layer_norm = keras.layers.LayerNormalization() # Create layer normalization layer
        super(LayerNormLSTMCell, self).__init__(**kwargs)
    
    def build(self, batch_input_shape):
        # Extract batch size
        self.batch_size = batch_input_shape[0]
        # Initially set weights batch size dimeninsion as 1, and repeat for each batch later - else we get batch_size*num weights parameters to train
        # Build weights for forget gate
        self.forget_w = self.add_weight(shape=(1, self.state_size[0], self.state_size[0] + batch_input_shape[-1]), initializer='glorot_uniform', name='forget_w')
        self.forget_b = self.add_weight(shape=(1, self.state_size[0],), initializer='glorot_uniform', name='forget_b')
        # Build weights for input gate
        self.input_g_w = self.add_weight(shape=(1, self.state_size[0], self.state_size[0] + batch_input_shape[-1]), initializer='glorot_uniform', name='input_g_w')
        self.input_g_b = self.add_weight(shape=(1, self.state_size[0],), initializer='glorot_uniform', name='input_g_b')
        self.input_i_w = self.add_weight(shape=(1, self.state_size[0], self.state_size[0] + batch_input_shape[-1]), initializer='glorot_uniform', name='input_i_w')
        self.input_i_b = self.add_weight(shape=(1, self.state_size[0],), initializer='glorot_uniform', name='input_i_b')
        # Build weights for output gate
        self.output_w = self.add_weight(shape=(1, self.state_size[0], self.state_size[0] + batch_input_shape[-1]), initializer='glorot_uniform', name='output_w')
        self.output_b = self.add_weight(shape=(1, self.state_size[0],), initializer='glorot_uniform', name='output_b')
        # Tell keras that this layer is built
        self.built = True
        
    def merge_with_state(self, inputs):
        # Concatenate hidden state from previous step, and inputs from current state
        # We can then apply weight transformations to both together for each gate step
        self.stateH = K.concatenate([self.stateH, inputs], axis=-1)

    def forget_gate(self):        
        # From 'f' controller
        # Repeat weights for full batch size here so 
        f_t = K.batch_dot(K.repeat_elements(self.forget_w, rep=self.batch_size, axis=0), self.stateH) + self.forget_b
        # Conduct layer norm before applying non-linearity
        if self.layer_normalization:
            f_t = self.layer_norm(f_t)
        # Perform sigmoid activation to layer normalized neurons
        f_t = K.sigmoid(f_t)
        return f_t

    def input_gate(self):
        # From 'g' controller - select candidates from short term state
        g_t = K.batch_dot(K.repeat_elements(self.input_g_w, rep=self.batch_size, axis=0), self.stateH) + self.input_g_b
        # Conduct layer norm before applying non-linearity
        if self.layer_normalization:
            g_t = self.layer_norm(g_t)
        # Perform tanh activation to layer normalized neurons
        g_t = K.tanh(g_t)
        # Apply recurrent dropout to the hidden state updates vector only
        # https://arxiv.org/pdf/1603.05118.pdf
        g_t = K.dropout(g_t, level=self.recurrent_dropout_rate, seed=None)
        
        # From 'i' controller - decide which candidates from g to keep in long term state
        i_t = K.batch_dot(K.repeat_elements(self.input_i_w, rep=self.batch_size, axis=0), self.stateH) + self.input_i_b
        # Conduct layer norm before applying non-linearity
        if self.layer_normalization:
            i_t = self.layer_norm(i_t)
        # Perform sigmoid activation to layer normalized neurons
        i_t = K.sigmoid(i_t)
        
        return g_t, i_t

    def output_gate(self):
        self.stateH = K.batch_dot(K.repeat_elements(self.output_w, rep=self.batch_size, axis=0), self.stateH) + self.output_b
        # Conduct layer norm before applying non-linearity
        if self.layer_normalization:
            self.stateH = self.layer_norm(self.stateH)
        self.stateH = K.sigmoid(self.stateH)
        # Conduct layer norm before applying non-linearity
        if self.layer_normalization:
            if self.layer_norm_c_state:
                self.stateC = self.layer_norm(self.stateC)
                
        # Add componenet from long term state
        self.stateH = self.stateH * K.tanh(self.stateC)
        
        # Apply dropout to outputs only
        # https://arxiv.org/pdf/1409.2329.pdf
        self.stateHOut = self.stateH #K.dropout(self.stateH, level=self.dropout_rate, seed=None)

    def call(self, inputs, states):
        
        # Perform dropout on inputs, and recurrent dropout on states
        #inputs = K.dropout(inputs, level=self.dropout_rate, seed=None)
        
        # Extract short term and long term states
        self.stateH = states[0]
        self.stateC = states[1]
        #
        self.merge_with_state(inputs)
        # Process forget gate
        self.f_t = self.forget_gate()
        # Process input gate
        self.g_t, self.i_t = self.input_gate()
        # Update long term state
        self.stateC = self.stateC * self.f_t + self.g_t * self.i_t
        # Process output gate
        self.output_gate()

        return self.stateHOut, [self.stateH, self.stateC]
    
    # Update config to ease loading and saving of custom layer. Note saving of full model with custom layer requires
    # Tensorflow 2.2.0 or later 
    def get_config(self):
        config = super(LayerNormLSTMCell, self).get_config()
        config.update({
            'units': self.units,
#           'batch_size':self.batch_size,
#             'state_size': self.state_size,
            'dropout_rate': self.dropout_rate,
            'recurrent_dropout_rate': self.recurrent_dropout_rate,
            'layer_normalization': self.layer_normalization,
            'layer_norm_c_state':self.layer_norm_c_state,
#             'layer_norm':self.layer_norm
        })
        return config
    

class LayerNormRNNCell(keras.layers.Layer):
    """ RNN Cell with Layer Normalization, based off of SimpleRNNCell.
    Apply as a layer using tensorflow.keras.layers.RNN(LayerNormRNNCell(), return_sequence=True...)
    : param units: number units in cell to pass as number of hidden state units and output units (same for RNN)
    : param layer_normalization: boolean flag determining whether or not to perform layer normalization on the features
    """
    def __init__(self, units, layer_normalization:bool=True, activation="tanh", **kwargs):
        super().__init__(**kwargs) # Initialize kwargs from parent layer
        self.state_size = units
        self.output_size = units
        # Create RNN cell. Note we do not call activation, as we want to apply Layer Norm before the activation
        self.simple_rnn_cell = keras.layers.SimpleRNNCell(units, activation=None) 
        # Flag for whether or not to apply layer normalization and create layer normalization layer
        self.layer_normalization = layer_normalization
        self.layer_norm = keras.layers.LayerNormalization()
        # Fetch activation function for cell
        self.activation = keras.activations.get(activation)
        
    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states) # Call RNN cell on current inputs and previous hidden states
        # Note outputs and new_states are the same for RNN so we just apply layer normalization to the outputs and ignore new_states
        if self.layer_normalization:
            norm_outputs = self.activation(self.layer_norm(outputs)) # Apply layer normalization layer to outputs, THEN apply activation
        else: # If not applying layer normalization, this becomes a SimpleRNNCEll
            norm_outputs = self.activation(outputs)
        return norm_outputs, [norm_outputs]