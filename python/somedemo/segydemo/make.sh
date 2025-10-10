#!/bin/bash
# -*- coding: utf-8 -*-
# ./config/nvim/init.vim
# This script is used to build the project.
# Author: Zifei Li
# Date: 2024-01-16 01:40
# File: make.sh
scons view
scons
python ./test.py
scons -c
