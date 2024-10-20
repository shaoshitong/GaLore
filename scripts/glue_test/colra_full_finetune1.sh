CUDA_VISIBLE_DEVICES=2 python /home/data02/zj23/GaLore-master/run_glue_cola_full_finetune.py \
    --model_name_or_path /home/data02/zj23/GaLore-master/results/galore/roberta_base/sst2 \
    --max_length 512 \
    --seed 1234 \
    --per_device_train_batch_size 16 \
    --update_proj_gap 2000 \
    --learning_rate 3e-5 \
    --num_train_epochs 30 \
    --output_dir /home/data02/zj23/GaLore-master/results/full_finetune/roberta_base/cola