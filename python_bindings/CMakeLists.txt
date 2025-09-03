pybind11_add_module(tencers 
	tencers.cc)
set_target_properties(tencers PROPERTIES CXX_STANDARD 17)
set_target_properties(tencers PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(tencers PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/python)
target_link_libraries(tencers PUBLIC Tencers ElasticRods MeshFEM intel_pybind_14_hack)