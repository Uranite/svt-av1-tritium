# Patch aom's aom_pop_var(), which writes to a variable named "var".

if(NOT DEFINED FILE OR FILE STREQUAL "")
    message(FATAL_ERROR "Pass -DFILE=<path to aom's build/cmake/compiler_tests.cmake>")
endif()

file(READ "${FILE}" content)

set(bug "set(var \${SAVED_\${var}} PARENT_SCOPE)")
set(fix "set(\${var} \${SAVED_\${var}} PARENT_SCOPE)")

string(FIND "${content}" "${fix}" fixed_pos)
if(NOT fixed_pos EQUAL -1)
    message(STATUS "aom_pop_var is already fixed; nothing to do")
    return()
endif()

string(FIND "${content}" "${bug}" bug_pos)
if(bug_pos EQUAL -1)
    message(FATAL_ERROR "aom_pop_var pattern not found in ${FILE}; aom moved on - update fix_aom_pop_var.cmake")
endif()

string(REPLACE "${bug}" "${fix}" content "${content}")
file(WRITE "${FILE}" "${content}")
message(STATUS "Patched aom_pop_var in ${FILE}")
