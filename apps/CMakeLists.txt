add_executable(optphot_cli optphot_main.cpp)
set_target_properties(optphot_cli PROPERTIES OUTPUT_NAME optphot)
target_link_libraries(optphot_cli PRIVATE optphot optphot_warnings)

if(WITH_CUDA)
  add_executable(optphot_compare optphot_compare.cpp)
  target_link_libraries(optphot_compare PRIVATE optphot optphot_warnings)
endif()
