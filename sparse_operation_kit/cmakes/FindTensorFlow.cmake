set(PYTHON "python3")

execute_process(
    COMMAND
        ${PYTHON} -c
        "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TF_COMPILE_FLAGS)

execute_process(
    COMMAND
        ${PYTHON} -c
        "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TF_LINK_FLAGS)

string(REGEX MATCH "(^-L.*\ )" TF_LINK_DIR ${TF_LINK_FLAGS})
string(REPLACE "-L" "" TF_LINK_DIR ${TF_LINK_DIR})
string(REPLACE " " "" TF_LINK_DIR ${TF_LINK_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorFlow 
    DEFAULT_MSG TF_LINK_DIR
)

if (TensorFlow_FOUND)
    add_definitions(-DEIGEN_USE_GPU)
    mark_as_advanced(TF_LINK_DIR TF_LINK_FLAGS)
    message(STATUS "TF LINK FLAGS = ${TF_LINK_FLAGS}")
    message(STATUS "TF link dir = ${TF_LINK_DIR}")
    message(STATUS "TF COMPILE FLAGS = ${TF_COMPILE_FLAGS}")
endif()