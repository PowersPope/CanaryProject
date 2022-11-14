#!/bin/bash


for FILE in $(ls ${1}*.csv);
do 
  NEW_FILE=$(echo $FILE | sed 's/&/_/')
  gsed 's/,/*/2g' $FILE | gsed 's/duration.\+/duration/g' > clean_${NEW_FILE##*/}
done  
