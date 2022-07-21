cmake_minimum_required(VERSION 3.17)

project(rk_stat LANGUAGES C CXX)
set(TRITON_ENABLE_LOGGING ON)
if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Release)
  set(CMAKE_BUILD_TYPE Debug)
endif()

add_executable(
    ${CMAKE_PROJECT_NAME}
   main.cc
)

target_link_libraries(
    ${CMAKE_PROJECT_NAME}
  PRIVATE
    rknn_api
)


install(
  FILES
  ${CMAKE_PROJECT_NAME}
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../install
  PERMISSIONS WORLD_EXECUTE OWNER_WRITE OWNER_READ GROUP_READ
)