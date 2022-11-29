CUDA_VISIBLE_DEVICES=4 python run.py \
--model_name_or_path log/para_1w_loss_vocab/checkpoint-1950 \
--do_eval \
--train_file data/ChID/train_para.json \
--validation_file data/ChID/valid_para.json \
--test_file data/ChID/test_para.json \
--metric_for_best_model eval_accuracy \
--load_best_model_at_end \
--learning_rate 5e-5 \
--evaluation_strategy epoch \
--num_train_epochs 5 \
--output_dir ./tmp/para \
--per_device_eval_batch_size 16 \
--per_device_train_batch_size 8 \
--seed 42 \
--max_seq_length 512 \
--warmup_ratio 0.1 \
--save_strategy epoch \
--report_to tensorboard \
--overwrite_output
