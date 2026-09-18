# Autoware temporal velocity smoother

This package provides `TemporalVelocitySmoother`, an out-of-tree
`autoware_trajectory_modifier` plugin that re-times a fixed path on the node's constant time grid.
It solves a fixed-size, warm-started triple-integrator QP over position, velocity, acceleration,
and jerk. Velocity, acceleration, jerk, stop-position, lateral-acceleration, steering-rate, and
terminal stopping constraints are softened with independently weighted slack variables.

The first output point is at `t = trajectory_time_step`; the ego state at `t = 0` is the fixed
initial condition. The plugin moves poses along the incoming path so pose, speed, acceleration, and
time remain mutually consistent. Consumers must therefore not assume that point poses are stable
across this plugin.

Use `launch/trajectory_modifier_with_temporal_smoother.launch.xml` to run the existing trajectory
modifier with the supplied pipeline. That pipeline disables `TrajectoryVelocityOptimizer` and
places this plugin after geometry optimization. Running both longitudinal smoothers is unsupported.

`path_end_policy` supports:

- `clamp_s` (default): constrain the horizon to the known path.
- `extrapolate`: linearly extend the final path segment.
- `shrink_horizon`: keep the QP fixed-size but omit repeated points after the path end.

The QP remains exactly `8N` variables by `8N` rows for every input. Candidate histories and solver
instances are kept separately by candidate index. If the solver fails, the plugin publishes an
integrated, jerk-limited fallback profile and still reports the trajectory as modified.
