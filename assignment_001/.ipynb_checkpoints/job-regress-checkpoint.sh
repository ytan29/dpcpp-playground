#!/bin/bash

echo
echo start: $(date "+%H%M%S.%3N")
echo

./knn-regress -d gpu

echo
echo stop:  $(date "+%H%M%S.%3N")
echo
