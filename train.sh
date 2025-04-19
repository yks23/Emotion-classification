export Model1='lstm'
export Model2='gru'
export Model3='cnn'
export Model4='mlp'
export Model5='bilstm'
export Model6='bigru'
export Model7='bert'
export output_dir='checkpoint-new/bilstm'
export batch_size=32
export learning_rate=0.00002
export num_epochs=20
export hidden_size=128

# python train.py --output_dir $output_dir \
#                 --model $Model5 \
#                 --batch_size $batch_size \
#                 --learning_rate $learning_rate \
#                 --num_epochs $num_epochs \
#                 --hidden_dim $hidden_size & \
# python train.py --output_dir 'checkpoint-new/bigru' \
#                 --model $Model6 \
#                 --batch_size $batch_size \
#                 --learning_rate $learning_rate \
#                 --num_epochs $num_epochs \
#                 --hidden_dim $hidden_size & \
python train.py --output_dir 'checkpoint-new/bert' \
                --model $Model7 \
                --batch_size $batch_size \
                --learning_rate 0.0002 \
                --num_epochs $num_epochs \
                --hidden_dim $hidden_size 
# python train.py --output_dir $output_dir \
#                 --model $Model2 \
#                 --batch_size $batch_size \
#                 --learning_rate $learning_rate \
#                 --num_epochs $num_epochs \
#                 --hidden_dim $hidden_size  & \
# python train.py --output_dir $output_dir \
#                 --model $Model3 \
#                 --batch_size $batch_size \
#                 --learning_rate $learning_rate \
#                 --num_epochs $num_epochs \
#                 --hidden_dim $hidden_size & \
# python train.py --output_dir 'checkpoint-new/mlp-64' \
#                 --model $Model4 \
#                 --batch_size $batch_size \
#                 --learning_rate $learning_rate \
#                 --num_epochs $num_epochs \
#                 --hidden_dim 64 & \
# python train.py --output_dir 'checkpoint-new/mlp-256' \
#                 --model $Model4 \
#                 --batch_size $batch_size \
#                 --learning_rate $learning_rate \
#                 --num_epochs $num_epochs \
#                 --hidden_dim 256 & \
# python train.py --output_dir 'checkpoint-new/mlp-512' \
#                 --model $Model4 \
#                 --batch_size $batch_size \
#                 --learning_rate $learning_rate \
#                 --num_epochs $num_epochs \
#                 --hidden_dim 512 & \
# python train.py --output_dir 'checkpoint-new/mlp-lr-1e-4' \
#                 --model $Model4 \
#                 --batch_size $batch_size \
#                 --learning_rate 0.0001 \
#                 --num_epochs $num_epochs \
#                 --hidden_dim $hidden_size & \
# python train.py --output_dir 'checkpoint-new/mlp-lr-1e-5' \
#                 --model $Model4 \
#                 --batch_size $batch_size \
#                 --learning_rate 0.00001 \
#                 --num_epochs $num_epochs \
#                 --hidden_dim $hidden_size & \
# python train.py --output_dir 'checkpoint-new/mlp-lr-2e-3' \
#                 --model $Model4 \
#                 --batch_size $batch_size \
#                 --learning_rate 0.002 \
#                 --num_epochs $num_epochs \
#                 --hidden_dim $hidden_size & \
# python train.py --output_dir 'checkpoint-new/mlp-bs4' \
#                 --model $Model4 \
#                 --batch_size 4 \
#                 --learning_rate $learning_rate \
#                 --num_epochs $num_epochs \
#                 --hidden_dim $hidden_size & \
# python train.py --output_dir 'checkpoint-new/mlp-bs8' \
#                 --model $Model4 \
#                 --batch_size 8 \
#                 --learning_rate $learning_rate \
#                 --num_epochs $num_epochs \
#                 --hidden_dim $hidden_size & \
# python train.py --output_dir 'checkpoint-new/mlp-bs2' \
#                 --model $Model4 \
#                 --batch_size 2 \
#                 --learning_rate $learning_rate \
#                 --num_epochs $num_epochs \
#                 --hidden_dim $hidden_size & \
# python train.py --output_dir 'checkpoint-new/mlp-bs1' \
#                 --model $Model4 \
#                 --batch_size 1 \
#                 --learning_rate $learning_rate \
#                 --num_epochs $num_epochs \
#                 --hidden_dim $hidden_size 