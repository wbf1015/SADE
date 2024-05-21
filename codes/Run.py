import torch
import numpy as np

from DataLoader.RandomSample import *
from EmbeddingManager.NaiveManager2 import *
from EmbeddingManager.KDManager import *
from EmbeddingManager.KDManagerComplEx import *
from KGES.TransE import *
from KGES.RotatE import *
from KGES.SimplE import *
from KGES.ComplEx import *
from Models.NaiveModel import *
from Models.KDModel import *
from Models.ContrastiveKDModel import *
from Models.ContrastiveKDModel2 import *
from Loss.SigmoidLossOrigin import *
from Loss.HuberLoss import *
from Loss.KDLoss import *
from Loss.ContrastiveLoss import *
from Loss.MarginContrastiveLoss import *
from Optim.Optim import *
from Pruners.Constant import *
from Pruners.SemAttention import *
from Pruners.CSemAttention import *
from Excuter.StandardExcuter import *
from utils import *

args = parse_args()
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
set_logger(args)

np.random.randn(args.seed)
torch.manual_seed(args.seed)

'''
声明数据集
'''
train_triples, valid_triples, test_triples, all_true_triples, nentity, nrelation = read_data(args)

train_dataloader_head = DataLoader(
    TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'head-batch'), 
    batch_size=args.batch_size,
    shuffle=True, 
    num_workers=max(1, args.cpu_num//2),
    collate_fn=TrainDataset.collate_fn
)
train_dataloader_tail = DataLoader(
    TrainDataset(train_triples, nentity, nrelation, args.negative_sample_size, 'tail-batch'), 
    batch_size=args.batch_size,
    shuffle=True, 
    num_workers=max(1, args.cpu_num//2),
    collate_fn=TrainDataset.collate_fn
)
test_dataloader_head = DataLoader(
    TestDataset(
        test_triples, 
        all_true_triples, 
        args.nentity, 
        args.nrelation, 
        'head-batch'
    ), 
    batch_size=args.test_batch_size,
    num_workers=max(1, args.cpu_num//2), 
    collate_fn=TestDataset.collate_fn
)
test_dataloader_tail = DataLoader(
    TestDataset(
        test_triples, 
        all_true_triples, 
        args.nentity, 
        args.nrelation, 
        'tail-batch'
    ), 
    batch_size=args.test_batch_size,
    num_workers=max(1, args.cpu_num//2), 
    collate_fn=TestDataset.collate_fn
)
logging.info('Successfully init TrainDataLoader and TestDataLoader')

'''
声明Excuter组件
'''
# KGE=TransE(margin=args.gamma)
KGE=RotatE(margin=args.gamma)
# KGE=RotatE(margin=args.gamma, embedding_range=6.0+2.0, embedding_dim=args.hidden_dim)
# KGE=SimplE(margin=args.gamma)
# KGE=ComplEx(margin=args.gamma)

ContrastiveLoss = MarginInfoNCELoss(args)
# ContrastiveLoss = InfoNCELoss(args)

entity_pruner=CSemAttention8(args, ContrastiveLoss)
relation_pruner=Constant()

if args.negative_adversarial_sampling is False:
    loss=SigmoidLossOrigin(adv_temperature=None, margin=args.gamma)
else:
    loss=SigmoidLossOrigin(adv_temperature=args.adversarial_temperature, margin=args.gamma)

# soft_loss = HuberLoss(adv_temperature = args.adversarial_temperature, margin = args.kd_gamma, KGEmargin = args.gamma)
# soft_loss = KDLoss(adv_temperature = args.adversarial_temperature, margin = args.kd_gamma, KGEmargin = args.gamma)
if args.negative_adversarial_sampling is False:
    soft_loss = KDLoss(adv_temperature = None, margin = args.kd_gamma, KGEmargin = args.gamma)
else:
    soft_loss = KDLoss(adv_temperature = args.adversarial_temperature, margin = args.kd_gamma, KGEmargin = args.gamma)



# embedding_manager=NaiveManager2(args)
embedding_manager=KDManager(args)
# embedding_manager=KDManagerComplEx(args)


trainDataloader =BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
testDataLoaders=[test_dataloader_head, test_dataloader_tail]

optimizer=None
scheduler=None

Excuter = StandardExcuter(
    KGE=KGE, 
    model=ContrastiveKDModel2,
    embedding_manager=embedding_manager, 
    entity_pruner=entity_pruner, relation_pruner=relation_pruner, 
    loss=loss,
    kdloss=soft_loss,
    ContrastiveLoss=None,
    finetuner=None,
    trainDataloader=trainDataloader, testDataLoaders=testDataLoaders,
    optimizer=optimizer, scheduler=scheduler,
    args=args,
)

Excuter.Run()
