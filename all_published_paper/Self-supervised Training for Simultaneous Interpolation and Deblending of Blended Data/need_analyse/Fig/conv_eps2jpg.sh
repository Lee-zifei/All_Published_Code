#!/bin/bash

ls * | while read file 
do 
    old_file_name=${file%%.vpl*}.eps
    tran_file_name=${file%%.vpl*}.jpg

    #if [[ $old_file_name =~ \.eps ]]; then
    if [ "${file##*.}" == "vpl" ]; then 
	vpconvert ${file} format=eps color=y bgcolor=w txscale=0.5  
        convert -quality 200  "${old_file_name}" "${tran_file_name}"
        #convert -resize 1500x1500  "${old_file_name}" -background white -flatten "${tran_file_name}"
    fi
done
