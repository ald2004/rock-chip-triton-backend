cmake_minimum_required(VERSION 3.17)

project(triton_rockchip LANGUAGES C CXX)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_SOURCE_DIR}/../install/backends/rk")
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
# message(${CMAKE_INSTALL_PREFIX})
#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
# GPU support is disabled by default because recommended backend
# doesn't use GPUs.
#

if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_BUILD_TYPE Debug)
endif()


#
# Dependencies
#
# rk backend depends on common backend core.
#
find_package(TritonCommon REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../BoeTriton/install/common)
find_package(TritonCore REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../BoeTriton/install/core)
find_package(TritonBackend REQUIRED PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../../BoeTriton/install/backend)
set(TRIRON_COMMON_INSTALL_PATH "../../BoeTriton/install/common")
set(TRIRON_BACKEND_INSTALL_PATH "../../BoeTriton/install/backend")
set(TRIRON_CORE_INSTALL_PATH "../../BoeTriton/install/core")

#
# The backend must be built into a shared library. Use an ldscript to
# hide all symbols except for the TRITONBACKEND API.
#
configure_file(src/librock-chip_Backend.ldscript librock-chip_Backend.ldscript COPYONLY)

add_library(
  ${CMAKE_PROJECT_NAME} SHARED
  src/rock-chip_backend.cc
)

# add_library(
#   TutorialRecommendedBackend::triton-recommended-backend ALIAS triton-recommended-backend
# )

target_include_directories(
    ${CMAKE_PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TRIRON_COMMON_INSTALL_PATH}/include
    ${TRIRON_BACKEND_INSTALL_PATH}/include
    ${TRIRON_CORE_INSTALL_PATH}/include
)

target_compile_features(${CMAKE_PROJECT_NAME} PRIVATE cxx_std_11)
target_compile_options(
    ${CMAKE_PROJECT_NAME} PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-unused-variable -Wno-type-limits -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc>
)
# target_compile_options(LIBRARY_NAME SCOPE "-Wno-unused-variable  -Wno-error")

target_link_libraries(
    ${CMAKE_PROJECT_NAME}
  PRIVATE
    rknn_api
    TritonCore::triton-core-serverapi   # from repo-core
    TritonCore::triton-core-backendapi  # from repo-core
    TritonCore::triton-core-serverstub  # from repo-core
    TritonBackend::triton-backend-utils    # from repo-backend
)

if(WIN32)
  set_target_properties(
    ${CMAKE_PROJECT_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_recommended
  )
else()
  set_target_properties(
    ${CMAKE_PROJECT_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME ${CMAKE_PROJECT_NAME}
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/librock-chip_Backend.ldscript
    LINK_FLAGS "-Wl,--version-script librock-chip_Backend.ldscript"
  )
endif()

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/${CMAKE_PROJECT_NAME})

install(
  TARGETS
    ${CMAKE_PROJECT_NAME}
  EXPORT
  ${CMAKE_PROJECT_NAME}-targets
  PERMISSIONS WORLD_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}
  
)

# install(
#   EXPORT
#     triton-recommended-backend-targets
#   FILE
#     TutorialRecommendedBackendTargets.cmake
#   NAMESPACE
#     TutorialRecommendedBackend::
#   DESTINATION
#     ${INSTALL_CONFIGDIR}
# )

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/rock-chip_BackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}Config.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}Config.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT ${CMAKE_PROJECT_NAME}-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_PROJECT_NAME}Targets.cmake
  NAMESPACE BoeRockChipBackend::
)

export(PACKAGE ${CMAKE_PROJECT_NAME})