import sys
import os
import logging
import shutil
import torch.nn.functional as F
import torch
import torch.nn as nn

CODEPATH = os.path.abspath(os.path.dirname(__file__))
CODEPATH = CODEPATH.rsplit('/', 1)[0]
sys.path.append(CODEPATH)

from Models.NaiveModel import *
from Models.StageLossModel import *
from Optim.Optim import *

class StandardExcuter(object):
    def __init__(self, 
                 KGE=None, 
                 model=None,
                 embedding_manager=None, 
                 entity_pruner=None, relation_pruner=None, 
                 loss=None,
                 kdloss=None,
                 ContrastiveLoss=None,
                 finetuner=None,
                 trainDataloader=None, testDataLoaders=None,
                 optimizer=None, scheduler=None,
                 args=None,
    ):
        self.args = args
        self.model = model(KGE=KGE, embedding_manager=embedding_manager, entity_pruner=entity_pruner, relation_pruner=relation_pruner, loss=loss, args=args)
        self.set_part(self.model, kdloss=kdloss, finetuner=finetuner, ContrastiveLoss=ContrastiveLoss)
        self.trainDataloader = trainDataloader
        self.testDataLoaders = testDataLoaders # 这个应该是一个list
        self.args = args
        if self.args.init_checkpoint != 'without':
            self.load_model(self.model, args)
            
        if self.args.cuda:
            self.model.cuda()
        
        self.optimizer = getoptimizer(args, filter(lambda p: p.requires_grad, self.model.parameters()))
        self.scheduler = getscheduler(args, self.optimizer, last_epoch=-1)


    def Run(self):
        training_loss = []
        # metric = self.test_model()
        for step in range(self.args.steps):
            
            loss = self.train_step()
            training_loss.append(loss)
            
            if step%self.args.log_steps==0:
                def calculate_metrics(logs, prefix=''):
                    metrics = {}
                    for metric in logs[0].keys():
                        metrics[metric] = sum(log[metric] for log in logs) / len(logs)
                    self.log_metrics(step, metrics)
                
                calculate_metrics(training_loss)
                training_loss = []
            
            if (step+1)%self.args.save_checkpoint_steps==0:
                self.save_model(self.model, self.args)
            
            if (step+1)%(self.args.test_per_steps+1)==0:
                metric = self.test_model()
                self.log_metrics(step, metric)
                self.save_model(self.model, self.args)
            
        
        metric = self.test_model()
        self.log_metrics(self.args.steps, metric)
        self.save_model(self.model, self.args)
        
        
    def train_step(self):
        self.optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(self.trainDataloader)
        if self.args.cuda:
            positive_sample, negative_sample, subsampling_weight = positive_sample.cuda(), negative_sample.cuda(), subsampling_weight.cuda()
        loss, loss_record = self.model((positive_sample, negative_sample), subsampling_weight, mode)
        loss.backward()
        self.optimizer.step()
        
        if hasattr(self.model,"FineTuner"):
            self.model.update_embedding()
        
        return loss_record

    def test_model(self):
        self.model.eval()
        with torch.no_grad():
            logs = []
            step = 0
            total_steps = sum([len(dataset) for dataset in self.testDataLoaders])
            for test_dataset in self.testDataLoaders:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    if self.args.cuda:
                        positive_sample, negative_sample, filter_bias = positive_sample.cuda(), negative_sample.cuda(), filter_bias.cuda()
                        
                    batch_size = positive_sample.size(0)
                    score = self.model.predict((positive_sample, negative_sample), mode)
                    score = score[:, 1:]
                    
                    #Explicitly sort all the entities to ensure that there is no test exposure bias
                    if self.model.KGE.margin is not None:
                        score += filter_bias
                        argsort = torch.argsort(score, dim = 1, descending=True) # 降序
                    else:
                        score -= filter_bias
                        argsort = torch.argsort(score, dim = 1, descending=False)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        #Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        #ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                            
                        logs.append({
                            'MRR': 1.0/ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % self.args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs])/len(logs)

        self.model.train()
        return metrics
    
    
    def log_metrics(self, step, metrics):
        for metric in metrics:
            logging.info('%s at step %d: %f' % (metric, step, metrics[metric]))
    
    
    def write_dict_to_txt(self, dictionary, file_path):
        with open(file_path, 'w') as file:
            for key, value in dictionary.items():
                file.write(f"{key}: {value}\n")
    
    
    def save_model(self, model, args):
        argparse_dict = vars(args)
        self.write_dict_to_txt(argparse_dict, os.path.join(args.save_path, 'config.json'))

        Runpy_path = CODEPATH + '/Run.py'
        myrunsh_path = os.path.abspath(os.path.dirname(__file__)).rsplit('/', 2)[0] + '/myrun.sh'
        runsh_path = os.path.abspath(os.path.dirname(__file__)).rsplit('/', 2)[0] + '/run.sh'
        
        # 文件列表和它们的新名称
        files_to_copy = [Runpy_path, myrunsh_path, runsh_path]
        new_names = [os.path.join(args.save_path, os.path.basename(f) + 'Store' + os.path.splitext(f)[1]) for f in files_to_copy]

        # 复制并重命名文件
        for original, new in zip(files_to_copy, new_names):
            shutil.copy2(original, new)  # 使用 copy2 以保留元数据
        
        # 把模型和embedding的值都存上
        torch.save({
            'model_state_dict': model.state_dict(),
            },
            os.path.join(args.save_path, 'checkpoint')
        )
    
    def load_model(self, model, args):
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        model.load_state_dict(checkpoint['model_state_dict'])
    
    
    
    def set_part(self, model, kdloss=None, finetuner=None, ContrastiveLoss=None):
        if kdloss is not None:
            model.set_kdloss(kdloss)
        if finetuner is not None:
            model.FineTuner = finetuner
        if ContrastiveLoss is not None:
            model.ContrastiveLoss = ContrastiveLoss
    
    
        
        
        