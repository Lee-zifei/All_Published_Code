/* estimate the interference according to the shot.
%    Copyright (C) 2025 Chengdu University of Technology.
%    Copyright (C) 2025 Zifei Li.
%    Filename：estimate_interference.h
%    Author：Zifei Li
%    Institute：Chengdu University of Technology
%    Email：2024010196@stu.cdut.edu.cn
%    Work：2025/07/03/
%   Function： estimate the interference according to the shot
%   time                    : 2025/07/03
%   Author: Zifei Li
%   IN        input         :          A continuous aliasing signal of a seismic source
%             maintime      :          Enter the excitation time series of the source
%             assistime     :          A continuous aliasing signal of the target source
%             dt            :          Sampling time

%   OUT       interference  :          The Pseudo-deblending records with maintime
#pragma once:
*    This program is free software: you can redistribute it and/or modify it
*    under the terms of the GNU General Public License as published by the Free
*    Software Foundation, either version 3 of the License, or an later version.
*/

#include <rsf.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int* indices;
    int size;
} Find_Time_Point;

Find_Time_Point find_bef(float* temp, int temp_size,int time_length)
{
    Find_Time_Point result;
    result.indices = (int*)malloc(temp_size * sizeof(int));  
    result.size = 0;  
    for (int i = 0; i < temp_size; ++i) 
    {
        if (temp[i] >= 0 && temp[i] < time_length) 
        {
            result.indices[result.size] = i;  
            result.size++;  
        }
    }
    result.indices = (int*)realloc(result.indices, result.size * sizeof(int));
    return result;
}

Find_Time_Point find_aft(float* temp, int temp_size,int time_length) 
{
    Find_Time_Point result;
    result.indices = (int*)malloc(temp_size * sizeof(int));  
    result.size = 0;  
    for (int i = 0; i < temp_size; ++i) 
    {
        if (temp[i] <= 0 && temp[i] > -time_length) 
        {
            result.indices[result.size] = i;  
            result.size++; 
        }
    }
    result.indices = (int*)realloc(result.indices, result.size * sizeof(int));
    return result;
}


void free_find_result(Find_Time_Point result) {
    free(result.indices);
}

int main(int argc, char* argv[]) 
{
    sf_init(argc, argv);
    sf_file in = NULL, out = NULL, maintime_file = NULL, assisttime_file = NULL;

    float **input;
    float **output;
    float *temp;
    float *maintime;
    float *assistime;
    float dt;
    int nt, nx;

    in = sf_input("in");
    out = sf_output("out");
    maintime_file = sf_input ("maintime_file");
    assisttime_file = sf_input ("assisttime_file");

    if (SF_FLOAT != sf_gettype(maintime_file) ) sf_error("Need int maintime time file");
    /*The excitation time series of the main source*/
    if (SF_FLOAT != sf_gettype(assisttime_file) ) sf_error("Need int assistime time file");
    /*The excitation time series of the assist source*/
    if (!sf_getfloat("dt",&dt)) dt=2e-3;
    /* Sampleing rate */
    
    sf_histint(in, "n1", &nt);
    sf_histint(in, "n2", &nx);
    sf_histint(in, "n2", &nx);

    maintime = sf_floatalloc(nx);
    assistime = sf_floatalloc(nx);

    sf_floatread(&maintime[0], nx, maintime_file);
    sf_floatread(&assistime[0], nx, assisttime_file);


    input = sf_floatalloc2(nt, nx);
    output = sf_floatalloc2(nt, nx);
    temp = sf_floatalloc(nx);

    sf_floatread(&input[0][0], nt * nx, in);
    memset(output[0], 0, sizeof(float) * nt * nx);

    // for (int ix = 0; ix < nx; ix++)
    // {
    //     sf_warning("error bef index\n");

    // }
    
    for (int ix = 0; ix < nx; ix++)
    {
        for (int ix1 = 0; ix1 < nx; ix1++)
        {
            temp[ix1] = (assistime[ix] - maintime[ix1]) / dt;
        }

        Find_Time_Point bef = find_bef(temp, nx, nt);
        Find_Time_Point aft = find_aft(temp, nx, nt);

        if (bef.size!=0)
        {
            for (int j = 0; j < bef.size; j++)
            {
                int bef_idx = bef.indices[j];
                int index_limit_bef = (int)floor((maintime[bef_idx] + nt * dt - assistime[ix]) / dt);

                if (index_limit_bef >= 0 && index_limit_bef < nt)
                {
                    for (int k = 0; k <= index_limit_bef; k++)
                    {
                        output[ix][k] += input[bef_idx][nt - index_limit_bef + k];
                        // printf("%f\n",output[ix][k]);
                    }
                }
            }
        }
        if (bef.size == 0)
        {
            sf_warning("error bef index\n");
        }

        if (aft.size != 0)
        {
            for (int j = 0; j < aft.size; j++)
            {
                int aft_idx = aft.indices[j];
                int start_index = (int)floor((assistime[ix] + nt * dt - maintime[aft_idx]) / dt);
                int index_limit_aft = nt - start_index;

                if (index_limit_aft >= 0 && index_limit_aft < nt)
                {
                    int k1 = 0;
                    for (int k = index_limit_aft; k < nt; k++)
                    {
                        output[ix][k] += input[aft_idx][k1];
                        k1=k1+1;
                    }
                }
            }
        }
        if (aft.size == 0)
        {
            printf("error aft index\n");
        }
        free_find_result(bef);
        free_find_result(aft);
    }

    // sf_putint(out, "n1", nt);
    // sf_putint(out, "n2", nx);
    // sf_putfloat(out, "d1", dt);

    sf_floatwrite(output[0], nt * nx, out);
    // for (int ix = 0; ix < nx; ix++)
    // {
    //     for (int ix1 = 0; ix1 < nt; ix1++)
    //     {
    //         printf("%f\n",output[ix][ix1]);
    //     }
    // }
    exit(0);
}

