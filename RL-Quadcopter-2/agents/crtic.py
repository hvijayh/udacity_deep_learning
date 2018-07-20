from keras import layers, models, optimizers
from keras import backend as K
from keras import regularizers
from keras.layers import LeakyReLU


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, dropout_rate):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.reg_l2_val = 1e-4
        self.leaky_relu_alpha = 1e-2
        self.state_size = state_size
        self.action_size = action_size
        self.dropout_rate = dropout_rate
        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=64, kernel_regularizer=regularizers.l2(self.reg_l2_val))(states)
        net_states = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(self.dropout_rate)(net_states)

        net_states = layers.Dense(units=128, kernel_regularizer=regularizers.l2(self.reg_l2_val))(net_states)
        net_states = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Dropout(self.dropout_rate)(net_states)

        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=64, kernel_regularizer=regularizers.l2(self.reg_l2_val))(actions)
        net_actions = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(self.dropout_rate)(net_actions)

        net_actions = layers.Dense(units=128, kernel_regularizer=regularizers.l2(self.reg_l2_val))(net_actions)
        net_actions = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Dropout(self.dropout_rate)(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.LeakyReLU(alpha=self.leaky_relu_alpha)(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values', kernel_regularizer=regularizers.l2(self.reg_l2_val))(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(inputs=[*self.model.input, K.learning_phase()], outputs=action_gradients)