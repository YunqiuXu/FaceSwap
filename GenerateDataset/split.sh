#!/bin/bash

mkdir train
mkdir val

count=0
for curr_img in `ls *.png`
do
    div=`echo $(($count%10))`
    if [ $div -eq 0 ]
    then
        echo "$curr_img is cv"
        mv "$curr_img" val
    else
        mv "$curr_img" train
    fi
    
    count=`expr $count + 1`
done
