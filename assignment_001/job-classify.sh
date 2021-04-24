#!/bin/bash

export OverrideDefaultFP64Settings=1 
export IGC_EnableDPEmulation=1 


echo
echo start: $(date "+%H%M%S.%3N")
echo

./knn-classify -d gpu

echo
echo stop:  $(date "+%H%M%S.%3N")
echo
