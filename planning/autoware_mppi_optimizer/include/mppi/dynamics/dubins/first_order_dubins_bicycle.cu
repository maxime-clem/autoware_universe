#include <mppi/dynamics/dubins/first_order_dubins_bicycle.cuh>

#include <cmath>

namespace
{
using S = FirstOrderDubinsBicycleParams::StateIndex;
using C = FirstOrderDubinsBicycleParams::ControlIndex;

__host__ __device__ void firstOrderDubinsBicycleDeriv(
  const FirstOrderDubinsBicycleParams & p, const float * state, const float * control,
  float * state_der)
{
  const float v = state[static_cast<int>(S::VEL_X)];
  const float yaw = state[static_cast<int>(S::YAW)];
  const float steer = state[static_cast<int>(S::STEER_ANGLE)];
  const float accel = state[static_cast<int>(S::ACCELERATION)];
  const float accel_cmd = control[static_cast<int>(C::ACCELERATION_CMD)];
  const float steer_cmd = control[static_cast<int>(C::STEER_CMD)];

  const float accel_tau = fmaxf(p.accel_time_constant, 1.0E-4F);
  const float steer_tau = fmaxf(p.steer_time_constant, 1.0E-4F);

  state_der[static_cast<int>(S::ACCELERATION)] = (accel_cmd - accel) / accel_tau;
  state_der[static_cast<int>(S::VEL_X)] = accel;
  state_der[static_cast<int>(S::YAW)] = (v / p.wheel_base) * tanf(steer);

  float sin_yaw = 0.0F;
  float cos_yaw = 0.0F;
  sincosf(yaw, &sin_yaw, &cos_yaw);
  state_der[static_cast<int>(S::POS_X)] = v * cos_yaw;
  state_der[static_cast<int>(S::POS_Y)] = v * sin_yaw;

  const float steer_dot = clampSteerRate(p, (steer_cmd - steer) / steer_tau);
  state_der[static_cast<int>(S::STEER_ANGLE)] = steer_dot;
}

__host__ __device__ void populateGeometricDerivatives(
  const FirstOrderDubinsBicycleParams & p, const float * state, const float * next_state,
  float * output, const float dt)
{
  using O = FirstOrderDubinsBicycleParams::OutputIndex;
  const float safe_dt = fmaxf(dt, 1.0E-4F);
  const float wheel_base = fmaxf(p.wheel_base, 1.0E-4F);
  const float velocity = state[static_cast<int>(S::VEL_X)];
  const float next_velocity = next_state[static_cast<int>(S::VEL_X)];
  const float steer = state[static_cast<int>(S::STEER_ANGLE)];
  const float next_steer = next_state[static_cast<int>(S::STEER_ANGLE)];
  const float lateral_acceleration = velocity * velocity * tanf(steer) / wheel_base;
  const float next_lateral_acceleration =
    next_velocity * next_velocity * tanf(next_steer) / wheel_base;

  output[static_cast<int>(O::STEER_RATE)] = (next_steer - steer) / safe_dt;
  output[static_cast<int>(O::LONGITUDINAL_JERK)] =
    (next_state[static_cast<int>(S::ACCELERATION)] - state[static_cast<int>(S::ACCELERATION)]) /
    safe_dt;
  output[static_cast<int>(O::LATERAL_JERK)] =
    (next_lateral_acceleration - lateral_acceleration) / safe_dt;
}
}  // namespace

template <class CLASS_T, class PARAMS_T>
FirstOrderDubinsBicycleImpl<CLASS_T, PARAMS_T>::FirstOrderDubinsBicycleImpl(cudaStream_t stream)
: Dynamics<CLASS_T, PARAMS_T>(stream)
{
  this->params_ = PARAMS_T();
}

template <class CLASS_T, class PARAMS_T>
FirstOrderDubinsBicycleImpl<CLASS_T, PARAMS_T>::FirstOrderDubinsBicycleImpl(
  PARAMS_T & params, cudaStream_t stream)
: Dynamics<CLASS_T, PARAMS_T>(params, stream)
{
}

template <class CLASS_T, class PARAMS_T>
void FirstOrderDubinsBicycleImpl<CLASS_T, PARAMS_T>::computeDynamics(
  const Eigen::Ref<const state_array> & state, const Eigen::Ref<const control_array> & control,
  Eigen::Ref<state_array> state_der)
{
  firstOrderDubinsBicycleDeriv(this->params_, state.data(), control.data(), state_der.data());
}

template <class CLASS_T, class PARAMS_T>
bool FirstOrderDubinsBicycleImpl<CLASS_T, PARAMS_T>::computeGrad(
  const Eigen::Ref<const state_array> &, const Eigen::Ref<const control_array> &, Eigen::Ref<dfdx>,
  Eigen::Ref<dfdu>)
{
  return false;
}

template <class CLASS_T, class PARAMS_T>
void FirstOrderDubinsBicycleImpl<CLASS_T, PARAMS_T>::updateState(
  const Eigen::Ref<const state_array> state, Eigen::Ref<state_array> next_state,
  Eigen::Ref<state_array> state_der, const float dt)
{
  next_state = state + state_der * dt;
  if (this->params_.prevent_reverse_velocity) {
    next_state(static_cast<int>(S::VEL_X)) = fmaxf(next_state(static_cast<int>(S::VEL_X)), 0.0F);
  }
  next_state(static_cast<int>(S::YAW)) =
    angle_utils::normalizeAngle(next_state(static_cast<int>(S::YAW)));
  next_state(static_cast<int>(S::STEER_ANGLE)) = fmaxf(
    fminf(next_state(static_cast<int>(S::STEER_ANGLE)), this->params_.max_steer_angle),
    -this->params_.max_steer_angle);
  next_state(static_cast<int>(S::ACCELERATION)) = fmaxf(
    fminf(next_state(static_cast<int>(S::ACCELERATION)), this->params_.max_accel),
    this->params_.min_accel);
}

template <class CLASS_T, class PARAMS_T>
void FirstOrderDubinsBicycleImpl<CLASS_T, PARAMS_T>::step(
  Eigen::Ref<state_array> state, Eigen::Ref<state_array> next_state,
  Eigen::Ref<state_array> state_der, const Eigen::Ref<const control_array> & control,
  Eigen::Ref<output_array> output, const float /*t*/, const float dt)
{
  this->computeStateDeriv(state, control, state_der);
  this->updateState(state, next_state, state_der, dt);
  this->stateToOutput(next_state, output);
  populateGeometricDerivatives(this->params_, state.data(), next_state.data(), output.data(), dt);
}

template <class CLASS_T, class PARAMS_T>
FirstOrderDubinsBicycleImpl<CLASS_T, PARAMS_T>::state_array
FirstOrderDubinsBicycleImpl<CLASS_T, PARAMS_T>::interpolateState(
  const Eigen::Ref<state_array> state_1, const Eigen::Ref<state_array> state_2, const float alpha)
{
  state_array result = (1.0F - alpha) * state_1 + alpha * state_2;
  result(static_cast<int>(S::YAW)) = angle_utils::interpolateEulerAngleLinear(
    state_1(static_cast<int>(S::YAW)), state_2(static_cast<int>(S::YAW)), alpha);
  return result;
}

template <class CLASS_T, class PARAMS_T>
__device__ void FirstOrderDubinsBicycleImpl<CLASS_T, PARAMS_T>::updateState(
  float * state, float * next_state, float * state_der, const float dt)
{
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
  for (int i = threadIdx.y; i < PARENT_CLASS::STATE_DIM; i += blockDim.y) {
    next_state[i] = state[i] + state_der[i] * dt;
    if (i == static_cast<int>(S::VEL_X) && this->params_.prevent_reverse_velocity) {
      next_state[i] = fmaxf(next_state[i], 0.0F);
    }
    if (i == static_cast<int>(S::YAW)) {
      next_state[i] = angle_utils::normalizeAngle(next_state[i]);
    }
    if (i == static_cast<int>(S::STEER_ANGLE)) {
      next_state[i] =
        fmaxf(fminf(next_state[i], this->params_.max_steer_angle), -this->params_.max_steer_angle);
    }
    if (i == static_cast<int>(S::ACCELERATION)) {
      next_state[i] = fmaxf(fminf(next_state[i], this->params_.max_accel), this->params_.min_accel);
    }
  }
}

template <class CLASS_T, class PARAMS_T>
__device__ void FirstOrderDubinsBicycleImpl<CLASS_T, PARAMS_T>::step(
  float * state, float * next_state, float * state_der, float * control, float * output,
  float * theta_s, const float /*t*/, const float dt)
{
  this->computeStateDeriv(state, control, state_der, theta_s);
  __syncthreads();
  this->updateState(state, next_state, state_der, dt);
  __syncthreads();
  this->stateToOutput(next_state, output);
  __syncthreads();
  if (threadIdx.y == 0) {
    populateGeometricDerivatives(this->params_, state, next_state, output, dt);
  }
  __syncthreads();
}

template <class CLASS_T, class PARAMS_T>
__device__ void FirstOrderDubinsBicycleImpl<CLASS_T, PARAMS_T>::computeDynamics(
  float * state, float * control, float * state_der, float *)
{
  firstOrderDubinsBicycleDeriv(this->params_, state, control, state_der);
}

template <class CLASS_T, class PARAMS_T>
void FirstOrderDubinsBicycleImpl<CLASS_T, PARAMS_T>::enforceConstraints(
  Eigen::Ref<state_array> state, Eigen::Ref<control_array> control)
{
  PARENT_CLASS::enforceConstraints(state, control);
  if (!this->params_.prevent_reverse_velocity) {
    return;
  }

  const int velocity_idx = static_cast<int>(S::VEL_X);
  const int acceleration_idx = static_cast<int>(C::ACCELERATION_CMD);
  const float velocity = state(velocity_idx);
  const float acceleration_command = control(acceleration_idx);
  if (velocity >= 0.0F && velocity + acceleration_command * PARAMS_T::kControlDt < 0.0F) {
    control(acceleration_idx) = -velocity / PARAMS_T::kControlDt;
  }
}

template <class CLASS_T, class PARAMS_T>
__device__ void FirstOrderDubinsBicycleImpl<CLASS_T, PARAMS_T>::enforceConstraints(
  float * state, float * control)
{
  PARENT_CLASS::enforceConstraints(state, control);
  if (threadIdx.y != 0 || !this->params_.prevent_reverse_velocity) {
    return;
  }

  const int velocity_idx = static_cast<int>(S::VEL_X);
  const int acceleration_idx = static_cast<int>(C::ACCELERATION_CMD);
  const float velocity = state[velocity_idx];
  const float acceleration_command = control[acceleration_idx];
  if (velocity >= 0.0F && velocity + acceleration_command * PARAMS_T::kControlDt < 0.0F) {
    control[acceleration_idx] = -velocity / PARAMS_T::kControlDt;
  }
}

template <class CLASS_T, class PARAMS_T>
void FirstOrderDubinsBicycleImpl<CLASS_T, PARAMS_T>::stateToOutput(
  const Eigen::Ref<const state_array> & state, Eigen::Ref<output_array> output)
{
  stateToOutput(state.data(), output.data());
}

template <class CLASS_T, class PARAMS_T>
__host__ __device__ void FirstOrderDubinsBicycleImpl<CLASS_T, PARAMS_T>::stateToOutput(
  const float * state, float * output)
{
  using O = FirstOrderDubinsBicycleParams::OutputIndex;
  const float v = state[static_cast<int>(S::VEL_X)];

  output[static_cast<int>(O::BASELINK_VEL_B_X)] = v;
  output[static_cast<int>(O::BASELINK_VEL_B_Y)] = 0.0F;
  output[static_cast<int>(O::BASELINK_POS_I_X)] = state[static_cast<int>(S::POS_X)];
  output[static_cast<int>(O::BASELINK_POS_I_Y)] = state[static_cast<int>(S::POS_Y)];
  output[static_cast<int>(O::YAW)] = state[static_cast<int>(S::YAW)];
  output[static_cast<int>(O::STEER_ANGLE)] = state[static_cast<int>(S::STEER_ANGLE)];
  output[static_cast<int>(O::ACCELERATION)] = state[static_cast<int>(S::ACCELERATION)];
  output[static_cast<int>(O::TOTAL_VELOCITY)] = fabsf(v);
  output[static_cast<int>(O::STEER_RATE)] = 0.0F;
  output[static_cast<int>(O::LONGITUDINAL_JERK)] = 0.0F;
  output[static_cast<int>(O::LATERAL_JERK)] = 0.0F;
}

template <class CLASS_T, class PARAMS_T>
FirstOrderDubinsBicycleImpl<CLASS_T, PARAMS_T>::state_array
FirstOrderDubinsBicycleImpl<CLASS_T, PARAMS_T>::stateFromMap(
  const std::map<std::string, float> & map)
{
  state_array s = state_array::Zero();
  const auto set_if = [&map, &s](const char * key, const int idx) {
    const auto it = map.find(key);
    if (it != map.end()) {
      s(idx) = it->second;
    }
  };
  set_if("VEL_X", static_cast<int>(S::VEL_X));
  set_if("YAW", static_cast<int>(S::YAW));
  set_if("POS_X", static_cast<int>(S::POS_X));
  set_if("POS_Y", static_cast<int>(S::POS_Y));
  set_if("STEER_ANGLE", static_cast<int>(S::STEER_ANGLE));
  set_if("ACCELERATION", static_cast<int>(S::ACCELERATION));
  return s;
}
