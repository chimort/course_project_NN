find_program(CLANG_FORMAT_BIN clang-format)

if(NOT CLANG_FORMAT_BIN)
    message(FATAL_ERROR "clang-format not found!")
endif()

# Получаем список измененных файлов с помощью git
execute_process(
    COMMAND git diff --name-only HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE CHANGED_FILES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Разделяем список файлов на отдельные строки
string(REPLACE "\n" ";" CHANGED_FILES_LIST "${CHANGED_FILES}")

set(FILES_TO_FORMAT "")
foreach(FILE ${CHANGED_FILES_LIST})
    if(FILE MATCHES "\\.(cpp|h)$" AND NOT FILE MATCHES "^external/")
        # Проверяем, существует ли файл
        if(EXISTS "${CMAKE_SOURCE_DIR}/${FILE}")
            list(APPEND FILES_TO_FORMAT "${CMAKE_SOURCE_DIR}/${FILE}")
        endif()
    endif()
endforeach()

if(FILES_TO_FORMAT)
    add_custom_target(clang-format
        COMMAND ${CLANG_FORMAT_BIN} -i ${FILES_TO_FORMAT}
        COMMENT "Running clang-format on changed source files"
    )
else()
    add_custom_target(clang-format
        COMMAND ${CMAKE_COMMAND} -E echo "No C++ source files have been changed."
    )
endif()