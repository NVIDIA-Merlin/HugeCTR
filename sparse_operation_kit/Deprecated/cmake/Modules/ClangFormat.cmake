# Copyright Tomas Zeman 2018.
# Distributed under the Boost Software License, Version 1.0.
# (See accompanying file LICENSE_1_0.txt or copy at
# http://www.boost.org/LICENSE_1_0.txt)

function(clangformat_setup clangformat_srcs)
  if(NOT CLANGFORMAT_EXECUTABLE)
    set(CLANGFORMAT_EXECUTABLE clang-format)
  endif()

  if(NOT EXISTS ${CLANGFORMAT_EXECUTABLE})
    find_program(clangformat_executable_tmp ${CLANGFORMAT_EXECUTABLE})
    if(clangformat_executable_tmp)
      set(CLANGFORMAT_EXECUTABLE ${clangformat_executable_tmp})
      unset(clangformat_executable_tmp)
    else()
      message(FATAL_ERROR "ClangFormat: ${CLANGFORMAT_EXECUTABLE} not found! Aborting")
    endif()
  endif()

  foreach(clangformat_src ${clangformat_srcs})
    get_filename_component(clangformat_src ${clangformat_src} ABSOLUTE)
    list(APPEND clangformat_srcs_tmp ${clangformat_src})
  endforeach()
  set(clangformat_srcs "${clangformat_srcs_tmp}")
  unset(clangformat_srcs_tmp)

  add_custom_target(${PROJECT_NAME}_clangformat
                    COMMAND ${CLANGFORMAT_EXECUTABLE}
                            -style=file
                            -i
                            ${clangformat_srcs}
                    COMMENT "Formating with ${CLANGFORMAT_EXECUTABLE} ...")

  if(TARGET clangformat)
    add_dependencies(clangformat ${PROJECT_NAME}_clangformat)
  else()
    add_custom_target(clangformat DEPENDS ${PROJECT_NAME}_clangformat)
  endif()
endfunction()
