cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build/ -j6
chmod +x build/CourseProject
./build/CourseProject