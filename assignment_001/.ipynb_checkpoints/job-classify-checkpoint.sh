#!/bin/bash

echo
echo start: $(date "+%y%m%d.%H%M%S.%3N")
echo

./knn-classify -d cpu

echo
echo stop:  $(date "+%y%m%d.%H%M%S.%3N")
echo