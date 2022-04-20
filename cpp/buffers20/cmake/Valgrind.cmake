find_program(MEMORYCHECK_COMMAND valgrind)
if(MEMORYCHECK_COMMAND)
	add_custom_target(memcheck
		COMMAND ${MEMORYCHECK_COMMAND} -q --tool=memcheck --trace-children=yes --leak-check=full $<TARGET_FILE:test_buffers>)
	add_custom_target(threadcheck
		COMMAND ${MEMORYCHECK_COMMAND} -q --tool=helgrind $<TARGET_FILE:test_buffers>)
endif()

