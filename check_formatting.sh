#!/bin/bash

# Функция для проверки форматирования с помощью black
check_black_formatting() {
    local file=$1
    echo "Checking formatting for: $file"
    black "$file"
    if [ $? -eq 0 ]; then
        echo "$file is formatted correctly by black."
    else
        echo "Formatting needed for: $file"
    fi
}

# Функция для проверки кодстайла и импортов с помощью ruff
check_ruff_linting() {
    local file=$1
    echo "Checking linting for: $file"
    ruff check "$file"
    if [ $? -eq 0 ]; then
        echo "$file passed ruff linting."
    else
        echo "Ruff linting issues found in: $file"
    fi
}

check_isort_formatting() {
    local file=$1
    echo "Checking and sorting imports (if needed) for: $file"
    isort "$file"
    if [ $? -eq 0 ]; then
        echo "$file imports are sorted correctly by isort."
    else
        echo "Isort reformatted imports in the file: $file"
    fi
}


for file in $(find . -type f -name "*.py" -not -path "./.venv/*"); do
    echo -e "\nChecking file: $file"

    # Проверка black
    check_black_formatting "$file"

    check_isort_formatting "$file"

    # Проверка ruff
    check_ruff_linting "$file"
done
