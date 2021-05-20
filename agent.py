import tensorflow as tf

from tf_agents.agents import data_converter
from tf_agents.agents import tf_agent
from tf_agents.agents.ppo import ppo_policy
from tf_agents.agents.ppo import ppo_utils
from tf_agents.networks import network
from tf_agents.policies import greedy_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory
from tf_agents.typing import types
from tf_agents.utils import common
from tf_agents.utils import eager_utils
from tf_agents.utils import nest_utils
from tf_agents.utils import object_identity
from tf_agents.utils import tensor_normalizer
from tf_agents.utils import value_ops

class tempAgent(tf_agent.TFAgent):

    def __init__(
        self,
        time_step_spec: ts.TimeStep,
        action_spec: types.NestedTensorSpec,
        optimizer: Optional[types.Optimizer] = None,
        policy_value_net: Optional[network.Network] = None,
        actionFesibility_net: Optional[network.Network] = None,
        lambda_value: types.Float = 0.95,
        discount_factor: types.Float = 0.99,
        policy_l2_reg: types.Float = 0.0,
        value_function_l2_reg: types.Float = 0.0,
        shared_vars_l2_reg: types.Float = 0.0,
        num_epochs: int = 25,
        use_td_lambda_return: bool = False,
        normalize_rewards: bool = True,
        reward_norm_clipping: types.Float = 10.0,
        normalize_observations: bool = True,
        gradient_clipping: Optional[types.Float] = None,
        value_clipping: Optional[types.Float] = None,
        check_numerics: bool = False,
        # TODO(b/150244758): Change the default to False once we move
        # clients onto Reverb.
        compute_value_and_advantage_in_train: bool = True,
        update_normalizers_in_train: bool = True,
        debug_summaries: bool = False,
        summarize_grads_and_vars: bool = False,
        train_step_counter: Optional[tf.Variable] = None,
        name: Optional[Text] = None):
    

        if not isinstance(actor_net, network.Network):
        raise TypeError(
            'actor_net must be an instance of a network.Network.')
        if not isinstance(value_net, network.Network):
        raise TypeError('value_net must be an instance of a network.Network.')

        
        policy_value_net.create_variables(time_step_spec.observation)
        actionFesibility_net.create_variables(time_step_spec.observation)

        tf.Module.__init__(self,name=name)

        #local variables
        self._policy_value_network = policy_value_net
        self._actionFeasibility_network = actionFesibility_net
        
        self._as_trajectory = data_converter.AsTrajectory(
            self.data_context, sequence_length=None)

    def _train(self, experience, weights):
        #exp = (time_step, pi, z, nextstate, T/F)
        experience = self._as_trajectory(experience)


    def _policy_value_loss(self, 
                        experience,
                        weights=None, 
                        training=False):
        
        transition = self._as_transition(experience)
        time_steps, policy_steps, next_time_steps = transition
        actions = policy_steps.action

        with tf.name_scope('pv_loss'):
            network_observation = time_steps.observation
            #input
            (p,v),network_state = self._policy_value_network(network_observation,
                                                            step_type=time_steps.step_type,
                                                            training=training)

            ce = tf.keras.losses.CategoricalCrossentropy()
            pi = None
            policy_loss = ce(pi,p)


            mse = tf.keras.losses.MeanSquaredError()
            value_loss = mse(z,v)

            total_loss = policy_loss + value_loss

            agg_loss = common.aggregate_losses(
                per_exapmle_loss = total_loss,
                sample_weight=weights,
                regularization_loss=self._policy_value_network.losses
            )
            total_loss = agg_loss.total_loss

        return tf_agent.LossInfo(total_loss)


    def _actionFesibility_loss(self,
                            experience,
                            weights=None,
                            training=False):

        transition = self._as_transition(experience)
        time_steps, policy_steps, next_time_steps = transition
        actions = policy_steps.action

        with tf.name_scope('af_loss'):
            network_observation = time_steps.observation

            pred, _ = self._actionFeasibility_network(network_observation,
                                                    step_type=time_steps.step_type,
                                                    training=training)

            mse = tf.keras.losses.MeanSquaredError()
            loss = mse(y_true,pred)

            agg_loss = common.aggregate_losses(
                per_exapmle_loss = loss,
                sample_weight=weights,
                regularization_loss=self._actionFeasibility_network.losses
            )
            total_loss = agg_loss.total_loss

        return tf_agent.LossInfo(total_loss)
