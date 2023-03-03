#! /bin/bash

# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# log_dir=log_mp1
# rm -rf $log_dir

# 345M mp1
# python  projects/gpt/inference.py --mp_degree 1 --model_dir output

# 345M mp2
python -u -m paddle.distributed.launch \
    --devices "0,1" \
    --log_dir "gpt345m_log" \
    projects/gpt/inference.py --model_dir "./output_gpt3_345M" --mp_degree 2

# 6.7B mp4
# python -u -m paddle.distributed.launch \
#     --devices "0,1,2,3" \
#     --log_dir "gpt6.7b_log" \
#     projects/gpt/inference.py --model_dir "./output" --mp_degree 4
