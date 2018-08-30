export PYTHONPATH="."

dataset_name=cifar10
dataset_n_classes=10
dataset_dir=datasets/${dataset_name}

dataloader_batch_size=32

network_name=resnet
network_args_1_n_classes=${dataset_n_classes}
network_args_2_nf_init=32
network_args_3_n_subsample=2
network_args_4_n_inter_block=5

for i in {1..5}
do
    experiment_folder=results/${dataset_name}/${network_name}/${network_args_1_n_classes}_${network_args_2_nf_init}_${network_args_3_n_subsample}_${network_args_4_n_inter_block}/try${i}

    python opts.py --dataloader-batch-size ${dataloader_batch_size} --dataset-name ${dataset_name} --dataset-n-classes ${dataset_n_classes} --dataset-dir ${dataset_dir} --network-name ${network_name} --network-args ${network_args_1_n_classes} ${network_args_2_nf_init} ${network_args_3_n_subsample} ${network_args_4_n_inter_block} --experiment-folder ${experiment_folder}

    python main.py ${experiment_folder}/opts.txt
done
