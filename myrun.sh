CODE_PATH=codes
DATA_PATH=data
SAVE_PATH=models
GPU_DEVICE=0

BATCH_SIZE=${1}
NEG_SAMPLE_SIZE=${2}
GAMMA=${3}
LR=${4}
STEPS=${5}
HIDDEN_DIM=${6}
TARGET_DIM=${7}
SEED=${8}
PRETRAIN_PATH=${9}
DATASET=${10}
ENTITY_MUL=${11}
RELATION_MUL=${12}
OPTIMIZER=${13}
SCHEDULER=${14}
TEST_BATCH_SIZE=${15}

DATA_PATH=$DATA_PATH/$DATASET
SAVE_PATH=$SAVE_PATH/"$DATASET"_"$SEED"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/Run.py -cuda \
    -seed $SEED \
    -data_path $DATA_PATH -save_path $SAVE_PATH -pretrain_path $PRETRAIN_PATH \
    -entity_mul $ENTITY_MUL -relation_mul $RELATION_MUL \
    -batch_size $BATCH_SIZE -negative_sample_size $NEG_SAMPLE_SIZE -gamma $GAMMA \
    -lr $LR -steps $STEPS \
    -hidden_dim $HIDDEN_DIM -target_dim $TARGET_DIM \
    -optimizer $OPTIMIZER -scheduler $SCHEDULER \
    -test_batch_size $TEST_BATCH_SIZE \
    ${16} ${17} ${18} ${19} ${20} ${21} ${22} ${23} ${24} ${25} ${26} \
    ${27} ${28} ${29} ${30} ${31} ${32} ${33} ${34} ${35} ${36} ${37} \
    ${38} ${39} ${40} ${41} ${42} ${43} ${44} ${45} ${46} ${47} ${48} \
