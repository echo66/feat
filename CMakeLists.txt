# Feat
# copyright 2017 William La Cava 
# license: GNU/GPL v3

# version requirement for Eigen
cmake_minimum_required (VERSION 3.0)
# use c++ 14
set (CMAKE_CXX_STANDARD 14)

# project name
project(Feat)

# add Eigen include paths
option(EIGEN_DIR "use Eigen environment variable" OFF)
if (EIGEN_DIR)
    include_directories("$ENV{EIGEN3_INCLUDE_DIR}")
else()
    find_package(Eigen3 REQUIRED NO_MODULE)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()

# add Shogun includes
# find Shogun library
#find_library(Shogun REQUIRED NO_MODULE)
option(SHOGUN_DIR "use environment Shogun paths" OFF)
if (SHOGUN_DIR)
    include_directories("$ENV{SHOGUN_DIR}")
    link_directories("$ENV{SHOGUN_LIB}")
else()
    find_package(Shogun REQUIRED NO_MODULE)
endif()

# Locate GTest
option(TEST "build tests" OFF)
if (TEST)
    option(GTEST_DIR "use GTest environment path" OFF)
    if (GTEST_DIR)
        include_directories("$ENV{GTEST_DIR}/include")

    else()
        find_package(GTest REQUIRED)
        include_directories(${GTEST_INCLUDE_DIRS})
    endif()
endif()

# autocomplete for YouCompleteMe needs commands to be exported
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set a default build type if none was specified
set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${default_build_type}' since none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g")
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# add OpenMP support
option(OMP "use OpenMP" ON)
if (OMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()
endif()

# gather list of source files
file(GLOB_RECURSE sources "src/*.cc")

list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/parser.cc)
list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc)
list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/cv_main.cc)

#feat library
add_library(feat_lib SHARED ${sources})

# executables

#feat executable
add_executable(feat src/main.cc)

#feat_cv executable
add_executable(feat_cv src/cv_main.cc) 

#link libraries for different executables
target_link_libraries(feat feat_lib shogun)
target_link_libraries(feat_cv feat_lib shogun)

if (TEST)
    # Link runTests with what we want to test and the GTest and pthread library
    add_executable(tests tests/gtest.cc)
    target_link_libraries(tests feat_lib shogun ${GTEST_LIBRARIES} pthread)
endif()


