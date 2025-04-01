python FireFlow-evolution_batch.py  \   
    --name 'flux-dev'     \
    --eval_dataset 'EditEval_v1'     \
    --feature_path 'feature'     \
    --guidance 2     \
    --num_steps 8     \
    --inject 1     \
    --start_layer_index 0     \
    --end_layer_index 37     \
    --output_dir 'examples/batch_edit-result'    \ 
    --output_prefix 'fireflow'     \
    --sampling_strategy 'fireflow'     \
    --offload     \
    --height 1024     \
    --width 1024     \
    --batch_size 1     \
    --num_workers 1     \
    --save_samples 

#需要安装 torchvision pandas torchmetrics openpyxl
#需要到 ./flux/util.py 里面设置 1.ckpt_path="/data/chx/FLUX.1-dev/flux1-dev.safetensors"   2.ae_path="/data/chx/FLUX.1-dev/ae.safetensors" （这两个路径改为你自己的safetensors文件的路径）