 
### mnist
python parameterBoard.py \
--client_select fix-frequency \
--batch_size 32 \
--lr 0.01 \
--gamma 0.998 \
--seed 1234 \
--num_nets 100 \
--part_nets_per_round 10 \
--fl_round 100 \
--dataname mnist \
--num_class 10 \
--model lenet \
--load_premodel True \
--save_model False \
--saved_model_name mnist_lenet_fl \
--partition_strategy homo \
--dir_parameter 0.5 \
--malicious_ratio 0.1 \
--backdoor_type trigger \
--trigger_type standard \
--untargeted_type none \
--trigger_label 0 \
--semantic_label 2 \
--poisoned_portion 0.3 \
--data_num 600 \
--manual_std 0.3 \
--device cuda:0 \
--defense_method none \
--file_name rlr


### fmnist
python parameterBoard.py \
--client_select fix-frequency \
--batch_size 32 \
--lr 0.01 \
--gamma 0.998 \
--seed 1234 \
--num_nets 100 \
--part_nets_per_round 10 \
--fl_round 100 \
--dataname fmnist \
--num_class 10 \
--model lenet \
--load_premodel True \
--save_model False \
--saved_model_name fmnist_lenet_fl \
--partition_strategy homo \
--dir_parameter 0.5 \
--malicious_ratio 0.3 \
--backdoor_type trigger \
--trigger_type manual \
--untargeted_type none \
--trigger_label 0 \
--semantic_label 2 \
--poisoned_portion 1 \
--data_num 600 \
--manual_std 0.3 \
--device cuda:0 \
--defense_method none \
--file_name fmnist

### cifar10
python parameterBoard.py \
--client_select fix-frequency \
--batch_size 32 \
--lr 0.00036 \
--gamma 0.998 \
--seed 1234 \
--num_nets 100 \
--part_nets_per_round 10 \
--fl_round 100 \
--dataname cifar10 \
--num_class 10 \
--model vgg9 \
--load_premodel True \
--save_model False \
--partition_strategy homo \
--dir_parameter 0.5 \
--malicious_ratio 0.3 \
--backdoor_type trigger \
--trigger_type manual \
--untargeted_type none \
--trigger_label 0 \
--semantic_label 2 \
--poisoned_portion 0.3 \
--data_num 500 \
--manual_std 0.08 \
--device cuda:0 \
--defense_method none \
--file_name aaa



### cifar100
python parameterBoard.py \
--client_select fix-frequency \
--batch_size 64 \
--lr 0.1 \
--gamma 0.998 \
--seed 1234 \
--num_nets 1 \
--part_nets_per_round 1 \
--fl_round 100 \
--local_training_epoch 1 \
--malicious_local_training_epoch 1 \
--dataname cifar100 \
--num_class 100 \
--model vgg16 \
--load_premodel False \
--save_model True \
--saved_model_name cifar100 \
--partition_strategy homo \
--dir_parameter 0.5 \
--malicious_ratio 0 \
--backdoor_type none \
--trigger_type standard \
--untargeted_type none \
--trigger_label 0 \
--semantic_label 2 \
--poisoned_portion 0.3 \
--data_num 100 \
--manual_std 0.08 \
--device cuda:0 \
--defense_method none \
--file_name cifar100
