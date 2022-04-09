if(NOT EXISTS "/cluster/project/infk/krause/andisheh/gorji/SHAP/swht_compress/install_manifest.txt")
  message(FATAL_ERROR "Cannot find install manifest: /cluster/project/infk/krause/andisheh/gorji/SHAP/swht_compress/install_manifest.txt")
endif()

file(READ "/cluster/project/infk/krause/andisheh/gorji/SHAP/swht_compress/install_manifest.txt" files)
string(REGEX REPLACE "\n" ";" files "${files}")
foreach(file ${files})
  message(STATUS "Uninstalling $ENV{DESTDIR}${file}")
  if(IS_SYMLINK "$ENV{DESTDIR}${file}" OR EXISTS "$ENV{DESTDIR}${file}")
    exec_program(
      "/cluster/apps/gcc-8.2.0/cmake-3.13.4-gqme7h75a7bivzoyn3vksxeolg6knx7k/bin/cmake" ARGS "-E remove \"$ENV{DESTDIR}${file}\""
      OUTPUT_VARIABLE rm_out
      RETURN_VALUE rm_retval
    )
    if(NOT "${rm_retval}" STREQUAL 0)
      message(FATAL_ERROR "Problem when removing $ENV{DESTDIR}${file}")
    endif()
  else(IS_SYMLINK "$ENV{DESTDIR}${file}" OR EXISTS "$ENV{DESTDIR}${file}")
    message(STATUS "File $ENV{DESTDIR}${file} does not exist.")
  endif()
endforeach()

set(includes_folder "/usr/local/include/swht")
exec_program(
  "/cluster/apps/gcc-8.2.0/cmake-3.13.4-gqme7h75a7bivzoyn3vksxeolg6knx7k/bin/cmake" ARGS "-E remove_directory \"$ENV{DESTDIR}${includes_folder}\""
  OUTPUT_VARIABLE rm_out
  RETURN_VALUE rm_retval
)
if(NOT "${rm_retval}" STREQUAL 0)
  message(FATAL_ERROR "Problem when removing $ENV{DESTDIR}${includes_folder}")
endif()
