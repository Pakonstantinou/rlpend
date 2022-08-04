cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(untitled6)
set(CMAKE_PREFIX_PATH "/home/pakos/libtorch/share/cmake/Torch")
# we ask for Magnum because we want to build the graphics
find_package(RobotDART REQUIRED OPTIONAL_COMPONENTS Magnum)
find_package (Threads)
find_package(Torch REQUIRED )
add_executable(robot_dart main.cpp)

set_property(TARGET robot_dart PROPERTY CXX_STANDARD 11)

target_link_libraries(robot_dart
        ${CMAKE_THREAD_LIBS_INIT}
        ${TORCH_LIBRARIES}
        RobotDART::Simu
        )

set_property(TARGET robot_dart PROPERTY CXX_STANDARD 11)


if(RobotDART_Magnum_FOUND)
    add_executable(robot_dart_graphics main.cpp)
    target_link_libraries(robot_dart_graphics
            ${CMAKE_THREAD_LIBS_INIT}
            RobotDART::Simu
            RobotDART::Magnum
            "${TORCH_LIBRARIES}"
            #${PYTHON_LIBRARIES}

            )
    set_property(TARGET robot_dart_graphics PROPERTY CXX_STANDARD 11)

endif()
