#!/bin/bash

FILE_NAME=FuzzyLDA-`date +%Y%m%d-%a`.tar.gz
tar -cvzf $FILE_NAME *.sh *.cpp CMakeLists.txt Python GDB
echo Done!

