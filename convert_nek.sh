#!/usr/bin/env bash

export DATA=`sed 's/^.*\"data\"\:{//' "$1" | sed 's/^/{/' | sed 's/}}$/},/'  | sed '1 i['`


echo $DATA \]| sed 's/, ]/]/'|  tee "$2" 
