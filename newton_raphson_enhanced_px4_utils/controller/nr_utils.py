import jax.numpy as jnp
from jax import jacfwd, lax

GRAVITY = 9.8  # Match Gazebo world
USING_CBFS = True
C = jnp.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1]])


def body2world_angular_rates(roll, pitch, body_rates):
    """Convert body angular rates [p, q, r] into world Euler angle rates."""
    transform = jnp.array([
        [1, jnp.sin(roll) * jnp.tan(pitch), jnp.cos(roll) * jnp.tan(pitch)],
        [0, jnp.cos(roll), -jnp.sin(roll)],
        [0, jnp.sin(roll) / jnp.cos(pitch), jnp.cos(roll) / jnp.cos(pitch)],
    ])
    return transform @ body_rates


def f_quad(state, input, mass):
    x, y, z, vx, vy, vz, roll, pitch, yaw = state
    del x, y, z
    curr_thrust = input[0]
    body_rates = input[1:]

    curr_rolldot, curr_pitchdot, curr_yawdot = body2world_angular_rates(roll, pitch, body_rates)

    sr = jnp.sin(roll)
    sy = jnp.sin(yaw)
    sp = jnp.sin(pitch)
    cr = jnp.cos(roll)
    cp = jnp.cos(pitch)
    cy = jnp.cos(yaw)

    vxdot = -(curr_thrust / mass) * (sr * sy + cr * cy * sp)
    vydot = -(curr_thrust / mass) * (cr * sy * sp - cy * sr)
    vzdot = GRAVITY - (curr_thrust / mass) * (cr * cp)

    return jnp.array([vx, vy, vz, vxdot, vydot, vzdot, curr_rolldot, curr_pitchdot, curr_yawdot])


def dynamics(state, input, mass):
    """Continuous-time quadrotor dynamics."""
    return f_quad(state, input, mass)


def interpolate_input(u_prev, u_next, progress, use_foh):
    """Select the control used during prediction."""
    progress = jnp.clip(progress, 0.0, 1.0)
    return lax.cond(
        use_foh,
        lambda _: u_prev + (u_next - u_prev) * progress,
        lambda _: u_next,
        operand=None,
    )


def rk4_pred(state, u_prev, u_next, lookahead_step, integrations_int, mass, use_foh):
    total_steps = jnp.maximum(1, integrations_int).astype(state.dtype)

    def input_at(stage_index, stage_fraction):
        progress = (stage_index.astype(state.dtype) + stage_fraction) / total_steps
        return interpolate_input(u_prev, u_next, progress, use_foh)

    def for_function(i, current_state):
        i_state = jnp.asarray(i, dtype=state.dtype)
        u_k1 = input_at(i_state, 0.0)
        u_k23 = input_at(i_state, 0.5)
        u_k4 = input_at(i_state, 1.0)

        k1 = dynamics(current_state, u_k1, mass)
        k2 = dynamics(current_state + k1 * lookahead_step / 2, u_k23, mass)
        k3 = dynamics(current_state + k2 * lookahead_step / 2, u_k23, mass)
        k4 = dynamics(current_state + k3 * lookahead_step, u_k4, mass)
        return current_state + (k1 + 2 * k2 + 2 * k3 + k4) * lookahead_step / 6

    return lax.fori_loop(0, integrations_int, for_function, state)


def predict_state(state, u_prev, u_next, T_lookahead, lookahead_step, mass, use_foh):
    """Predict the state at time t + T."""
    integrations_int = (T_lookahead / lookahead_step).astype(int)
    return rk4_pred(state, u_prev, u_next, lookahead_step, integrations_int, mass, use_foh)


def predict_output(state, u_prev, u_next, T_lookahead, lookahead_step, mass, use_foh):
    """Project the predicted state into the controlled output space."""
    pred_state = predict_state(state, u_prev, u_next, T_lookahead, lookahead_step, mass, use_foh)
    return C @ pred_state


def get_jac_pred_u(state, last_input, candidate_input, T_lookahead, lookahead_step, mass, use_foh):
    """Get the Jacobian of predicted output with respect to the candidate control."""
    raw_val = jacfwd(predict_output, 2)(
        state, last_input, candidate_input, T_lookahead, lookahead_step, mass, use_foh
    )
    return raw_val.reshape((4, 4))


def get_inv_jac_pred_u(state, last_input, candidate_input, T_lookahead, lookahead_step, mass, use_foh):
    """Get the pseudo-inverse Jacobian of predicted output with respect to control."""
    return jnp.linalg.pinv(
        get_jac_pred_u(
            state,
            last_input,
            candidate_input,
            T_lookahead,
            lookahead_step,
            mass,
            use_foh,
        ).reshape((4, 4))
    )


def execute_cbf(current, phi, max_value, min_value, gamma, switch_value=0.0):
    """Execute the control barrier function."""
    zeta_max = gamma * (max_value - current) - phi
    zeta_min = gamma * (min_value - current) - phi
    return jnp.where(
        current >= switch_value,
        jnp.minimum(0, zeta_max),
        jnp.maximum(0, zeta_min),
    )


def get_integral_cbf(last_input, phi):
    """Integral control barrier function setup for all inputs."""
    curr_thrust, curr_roll_rate, curr_pitch_rate, curr_yaw_rate = last_input
    phi_thrust, phi_roll_rate, phi_pitch_rate, phi_yaw_rate = phi

    thrust_gamma = 1.0
    thrust_max = 27.0
    thrust_min = 15.0
    switch_value = (thrust_max + thrust_min) / 2.0
    v_thrust = execute_cbf(curr_thrust, phi_thrust, thrust_max, thrust_min, thrust_gamma, switch_value)

    rates_max_abs = 0.8
    rates_max = rates_max_abs
    rates_min = -rates_max_abs
    gamma_rates = 1.0

    v_roll = execute_cbf(curr_roll_rate, phi_roll_rate, rates_max, rates_min, gamma_rates)
    v_pitch = execute_cbf(curr_pitch_rate, phi_pitch_rate, rates_max, rates_min, gamma_rates)
    v_yaw = execute_cbf(curr_yaw_rate, phi_yaw_rate, rates_max, rates_min, gamma_rates)

    return jnp.array([v_thrust, v_roll, v_pitch, v_yaw])


def get_enhanced_error(dgdx, rdot, state, candidate_input, mass):
    """Enhanced error term using reference-rate feedforward and state Jacobian coupling."""
    return rdot - dgdx @ dynamics(state, candidate_input, mass)


def get_jac_pred_x_uinv(state, last_input, candidate_input, T_lookahead, lookahead_step, mass, use_foh):
    """Get Jacobians of predicted output with respect to state and control."""
    jacobian_x, jacobian_u = jacfwd(predict_output, (0, 2))(
        state,
        last_input,
        candidate_input,
        T_lookahead,
        lookahead_step,
        mass,
        use_foh,
    )
    dgdu_inv = jnp.linalg.pinv(jacobian_u.reshape((4, 4)))
    return jacobian_x.reshape((4, 9)), dgdu_inv


def quaternion_from_yaw(yaw):
    """Convert a yaw angle to a quaternion."""
    half_yaw = yaw / 2.0
    return jnp.array([jnp.cos(half_yaw), 0, 0, jnp.sin(half_yaw)])


def quaternion_conjugate(q):
    """Return the conjugate of a quaternion."""
    return jnp.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def yaw_error_from_quaternion(q):
    """Return the yaw error from a quaternion angular error."""
    return 2 * jnp.arctan2(q[3], q[0])


def quaternion_normalize(q):
    """Normalize a quaternion."""
    return q / jnp.linalg.norm(q)


def shortest_path_yaw_quaternion(current_yaw, desired_yaw):
    """Return the shortest-path yaw error using quaternion arithmetic."""
    q_current = quaternion_normalize(quaternion_from_yaw(current_yaw))
    q_desired = quaternion_normalize(quaternion_from_yaw(desired_yaw))
    q_error = quaternion_multiply(q_desired, quaternion_conjugate(q_current))
    q_error_normalized = quaternion_normalize(q_error)
    return yaw_error_from_quaternion(q_error_normalized)


def get_tracking_error(ref, pred):
    """Calculate tracking error with wrapped yaw error."""
    err = ref - pred
    return err.at[3].set(shortest_path_yaw_quaternion(pred[3], ref[3]))
