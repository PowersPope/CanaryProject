#!/bin/bash


for FILE in $(ls ${1}*.csv);
do 
  gsed 's/,/*/2g' $FILE | gsed 's/duration.\+/duration/g' > ${FILE}_clean
done  
