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

log_dir=log_ernie10b_decoder
rm -rf $log_dir

export BKCL_PCIE_RING=1
# python -u -m paddle.distributed.launch \
#     --devices "2,3" \
#     --log_dir $log_dir \
#     projects/ernie/inference_ernie10b_decoder.py --model_dir "./models/ernie_decoder/" --mp_degree 2

python -u -m paddle.distributed.launch \
    --devices "0,3" \
    --log_dir $log_dir \
    projects/ernie/inference_ernie10b_decoder.py --model_dir "./models/ernie_decoder_fused/" --mp_degree 2
