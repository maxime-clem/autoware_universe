cmake_minimum_required(VERSION 3.14)
project(autoware_temporal_velocity_smoother)

find_package(autoware_cmake REQUIRED)
autoware_package()

find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

generate_parameter_library(temporal_velocity_smoother_parameters
  param/temporal_velocity_smoother_parameters.yaml
)

pluginlib_export_plugin_description_file(
  autoware_trajectory_modifier trajectory_modifier_plugins.xml)

ament_auto_add_library(autoware_temporal_velocity_smoother_core SHARED
  src/longitudinal_qp.cpp
  src/reference_path.cpp
  src/velocity_envelope.cpp
  src/warm_start.cpp
)
target_link_libraries(autoware_temporal_velocity_smoother_core Eigen3::Eigen)

ament_auto_add_library(autoware_temporal_velocity_smoother_plugin SHARED
  src/temporal_velocity_smoother_plugin.cpp
)
target_link_libraries(autoware_temporal_velocity_smoother_plugin
  autoware_temporal_velocity_smoother_core
  temporal_velocity_smoother_parameters
  ${autoware_trajectory_modifier_LIBRARIES}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_temporal_velocity_smoother
    test/test_longitudinal_qp.cpp
    test/test_reference_path.cpp
    test/test_plugin_integration.cpp
    test/test_velocity_envelope.cpp
    test/test_warm_start.cpp
  )
  target_link_libraries(test_temporal_velocity_smoother
    autoware_temporal_velocity_smoother_core
  )
endif()

ament_auto_package(INSTALL_TO_SHARE
  config
  launch
)
