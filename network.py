import gin
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.networks import encoding_network
from tf_agents.networks import network

class PolicyValueNetwork(network.Network):

    def __init__(self,
               input_tensor_spec,
               output_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               fc_layer_params=(75, 40),
               dropout_layer_params=None,
               activation_fn=tf.keras.activations.relu,
               kernel_initializer=None,
               batch_squash=True,
               dtype=tf.float32,
               name='PolicyValueNetwork'
    ):

        super(ValueNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        if not kernel_initializer:
        kernel_initializer = tf.compat.v1.keras.initializers.glorot_uniform()

        self._encoder = encoding_network.EncodingNetwork(
            input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            dtype=dtype)

        self._dense_layers = tf.keras.Sequential([
            tf.keras.layer.Dense(256,activation='relu'),
            tf.keras.layer.Dense(128,activation='relu'),
            tf.keras.layer.Dense(64,activation='relu')
        ])

        self._policy_layer = tf.keras.layer.Dense(output_spec,activations='softmax')
        self._value_layer = tf.keras.layer.Dense(1,activations='linear')

    def call(self, observation, step_type=None, network_state=(), training=False):
        state, network_state = self._encoder(
            observation, step_type=step_type, network_state=network_state,
            training=training
        )
        
        logits = self._dense_layers(state,training=training)

        policy = self._policy_layer(logits)
        value = self._value_layer(logits)

        return (policy,value), network_state


class ActionFeasiblityNetwork(network.Network):
    def __init__(self,
            input_tensor_spec,
            preprocessing_layers=None,
            preprocessing_combiner=None,
            conv_layer_params=None,
            fc_layer_params=(75, 40),
            dropout_layer_params=None,
            activation_fn=tf.keras.activations.relu,
            kernel_initializer=None,
            batch_squash=True,
            dtype=tf.float32,
            name='ActionFeasiblityNetwork'
    ):

        super(ValueNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)

        if not kernel_initializer:
        kernel_initializer = tf.compat.v1.keras.initializers.glorot_uniform()

        self._encoder = encoding_network.EncodingNetwork(
            input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            fc_layer_params=fc_layer_params,
            dropout_layer_params=dropout_layer_params,
            activation_fn=activation_fn,
            kernel_initializer=kernel_initializer,
            batch_squash=batch_squash,
            dtype=dtype)

        self._dense_layers = tf.keras.Sequential([
            tf.keras.layer.Dense(256,activation='relu'),
            tf.keras.layer.Dense(128,activation='relu'),
            tf.keras.layer.Dense(64,activation='relu')
        ])
        self._value_layer = tf.keras.layer.Dense(1,activations='linear')

    def call(self, observation, step_type=None, network_state=(), training=False):
        state, network_state = self._encoder(
            observation, step_type=step_type, network_state=network_state,
            training=training
        )
        
        logits = self._dense_layers(state,training=training)
        value = self._value_layer(logits)

        return value, network_state