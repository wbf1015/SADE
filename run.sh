bash myrun.sh 1024 256 6.0 0.0005 10000 512 32 3562 models/RotatE_FB15k-237_0 FB15k-237 2 1 Adam MultiStepLR 16 \
    -negative_adversarial_sampling -warm_up_steps 10000 \
    -token1 2 -head1 2 -head2 2 -t_dff 2 \
    --kd_gamma 4.0 --kdloss_weight 0.01 \
    -ckdloss_weight 0.01 -temperature 0.1 -contrastive_gamma 8.0 -ckdloss_dropout 0.1
     
