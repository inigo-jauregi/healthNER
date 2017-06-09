cmake_minimum_required(VERSION 2.6)
#PUBLIC VERSION
ADD_DEFINITIONS (-D_PUBLIC)

project("hcrfTest")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pg")
SET(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/../..)
# This is for gnu c++ compiler.
if(OPENMP GREATER 0)
  message("Compiling with openmp support")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

if(UNIX)
  ADD_DEFINITIONS(-DUNIX)
endif()

ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/../../libs/shared/hCRF hCRF)
INCLUDE_DIRECTORIES(${hCRF_SOURCE_DIR}/include)
ADD_EXECUTABLE("hcrfTest" "src/main.cpp")
LINK_DIRECTORIES(${hCRF_BINARY_DIR}/lib)

# All the library we need
TARGET_LINK_LIBRARIES("hcrfTest" hCRF)

TARGET_LINK_LIBRARIES("hcrfTest" cgDescent)

# FIND_LIBRARY(LIB_VAR3 "lbfgs" "../../libs/3rdParty/liblbfgs/lib/.libs/")
TARGET_LINK_LIBRARIES("hcrfTest" lbfgs)
# FIND_LIBRARY(LIB_VAR5 "uncoptim" "../../libs/3rdParty/uncOptim/lib")
TARGET_LINK_LIBRARIES("hcrfTest" uncoptim)


INSTALL(TARGETS hcrfTest 
  RUNTIME DESTINATION bin)

