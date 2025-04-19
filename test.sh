export Model='mlp'
export output_dir='result-mlp/9'
export hidden_size=128
export test_data='./Dataset/test.txt'
export model_path='./checkpoint-new/cnn_model_9.pth'
python test.py --output_dir $output_dir \
                --model $Model \
                --hidden_dim $hidden_size \
                --test_data  $test_data \
                --model_checkpoint $model_path \

export Model='lstm'
export output_dir='result-lstm/9'
export hidden_size=128
export test_data='./Dataset/test.txt'
export model_path='./checkpoint-new/lstm_model_9.pth'
python test.py --output_dir $output_dir \
                --model $Model \
                --hidden_dim $hidden_size \
                --test_data  $test_data \
                --model_checkpoint $model_path \

export Model='gru'
export output_dir='result-gru/9'
export hidden_size=128
export test_data='./Dataset/test.txt'
export model_path='./checkpoint-new/gru_model_9.pth'
python test.py --output_dir $output_dir \
                --model $Model \
                --hidden_dim $hidden_size \
                --test_data  $test_data \
                --model_checkpoint $model_path \

export Model='mlp'
export output_dir='result-mlp/9'
export hidden_size=128
export test_data='./Dataset/test.txt'
export model_path='./checkpoint-new/mlp/mlp_model_9.pth'
python test.py --output_dir $output_dir \
                --model $Model \
                --hidden_dim $hidden_size \
                --test_data  $test_data \
                --model_checkpoint $model_path \

