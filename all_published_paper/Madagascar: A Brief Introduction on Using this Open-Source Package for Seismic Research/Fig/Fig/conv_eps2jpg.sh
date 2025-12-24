#!/bin/bash

ls * | while read file 
do 
    old_file_name=${file}
    tran_file_name=${old_file_name%%.png*}.pdf

    if [[ $old_file_name =~ \.png ]]; then 
        #vpconvert $old_file_name format=eps color=y
        convert ${old_file_name} ${tran_file_name}
    fi

#    if [[ $old_file_name =~\.vpl ]]: then 
        
#    fi
done

