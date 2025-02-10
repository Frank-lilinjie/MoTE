import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import MoeNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
import copy

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = MoeNet(args, True)
        
        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]

        self.use_exemplars = args["use_old_data"]
        self.use_init_ptm = args["use_init_ptm"]
        self.use_diagonal = args["use_diagonal"]
        
        self.recalc_sim = args["recalc_sim"]
        self.alpha = args["alpha"] # forward_reweight is divide by _cur_task
        self.beta = args["beta"]

        self.moni_adam = args["moni_adam"]
        self.adapter_num = args["adapter_num"]
        self.cur_adapter = 1

        if self.moni_adam:
            self.use_init_ptm = True
            self.alpha = 1 
            self.beta = 1

    def after_task(self):
        self._known_classes = self._total_classes
        
    
    def get_cls_range(self, task_id):
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc
        
        return start_cls, end_cls
        
    # (proxy_fc = cls * dim)
    def replace_fc(self, train_loader):
        model = self._network
        model = model.eval()
        
        with torch.no_grad():           
            # replace proto for each adapter in the current task
            if self.use_init_ptm:
                start_idx = -1
            else:
                start_idx = 0

            for index in range(start_idx, self._cur_task + 1):
                embedding_list, label_list = [], []
                for i, batch in enumerate(train_loader):
                    (_, data, label) = batch
                    data = data.to(self._device)
                    label = label.to(self._device)
                    embedding = model.backbone.forward_proto(data, adapt_index=index)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())
                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)
                class_list = np.unique(self.train_dataset_for_protonet.labels)
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    if self.use_init_ptm:
                        model.fc.weight.data[class_index, :] = proto
                    else:
                        model.fc.weight.data[class_index, :] = proto
                model.backbone.update_proto_list(self._total_classes, model.fc.weight.data)
    
    def replace_fc_with_limited_adapters(self, train_loader):
        model = self._network
        model = model.eval()
        
        # 获取 Adapter 数量限制
        adapter_num = min(self._cur_task + 1, self.adapter_num)

        with torch.no_grad():
            # 提取前 adapter_num 个 Adapter 的类原型
            proto_by_adapter = {}
            for index in range(adapter_num):
                embedding_list, label_list = [], []

                for _, data, label in train_loader:
                    data = data.to(self._device)
                    label = label.to(self._device)
                    embedding = model.backbone.forward_proto(data, adapt_index=index)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())

                # 合并嵌入和标签
                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)

                # 提取每个类别的类原型
                class_list = label_list.unique().tolist()
                proto_by_adapter[index] = {}
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)  # 计算类原型
                    proto_by_adapter[index][class_index] = proto

            # 对于后续任务，通过加权混合计算类原型
            for class_index in range(self._total_classes):
                # 收集所有 Adapter 对该类别的类原型组成候选
                proto_candidates = []
                for index in range(adapter_num):
                    if class_index in proto_by_adapter[index]:
                        proto_candidates.append((proto_by_adapter[index][class_index], index))
                
                # 如果当前类别没有候选类原型，跳过（可能是新类别）
                if not proto_candidates:
                    continue

                # 计算每个候选类原型的权重
                weights = []
                for proto, index in proto_candidates:
                    if index == 0:  # 对应的历史类范围
                        ref_classes = list(range(0, self.adapter_num * self.inc))  # 假设每任务 10 类
                    else:
                        ref_classes = list(range(index * self.inc), (index + 1) * self.inc)

                    # 计算相似度
                    ref_protos = torch.stack([proto_by_adapter[index].get(c, torch.zeros_like(proto)) for c in ref_classes])
                    sim_scores = torch.cosine_similarity(proto.unsqueeze(0), ref_protos, dim=-1)
                    weight = sim_scores.mean().item()  # 平均相似度作为权重
                    weights.append(weight)

                # 归一化权重
                weights = torch.tensor(weights)
                weights = weights / weights.sum()

                # 融合类原型
                final_proto = sum(weight * proto_candidates[i][0] for i, weight in enumerate(weights))
                model.fc.weight.data[class_index, :] = final_proto

            # 更新模型中的类原型列表
            model.backbone.update_proto_list(self._total_classes, model.fc.weight.data)


         
    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        # self._network.show_trainable_params()
        
        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        
        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        
        self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),source="train", mode="test")
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
        self.replace_fc(self.train_loader_for_protonet)
        
        if self.cur_adapter < self.adapter_num:
            self._network.backbone.add_adapter_to_list(add_adapter = True)
            self.cur_adapter += 1
        else:
            self._network.backbone.add_adapter_to_list(add_adapter = False)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        
        if self._cur_task == 0 or self.init_cls == self.inc:
            optimizer = self.get_optimizer(lr=self.args["init_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])
        else:
            # for base 0 setting, the later_lr and later_epochs are not used
            # for base N setting, the later_lr and later_epochs are used
            if "later_lr" not in self.args or self.args["later_lr"] == 0:
                self.args["later_lr"] = self.args["init_lr"]
            if "later_epochs" not in self.args or self.args["later_epochs"] == 0:
                self.args["later_epochs"] = self.args["init_epochs"]

            optimizer = self.get_optimizer(lr=self.args["later_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["later_epochs"])

        self._init_train(train_loader, test_loader, optimizer, scheduler)
    
    def get_optimizer(self, lr):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                momentum=0.9, 
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr, 
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr, 
                weight_decay=self.weight_decay
            )

        return optimizer
    
    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        if self.moni_adam:
            if self._cur_task > self.adapter_num - 1:
                return
        
        if self._cur_task == 0 or self.init_cls == self.inc:
            epochs = self.args['init_epochs']
        else:
            epochs = self.args['later_epochs']
        
        prog_bar = tqdm(range(epochs))
            
        for _, epoch in enumerate(prog_bar):
            self._network.train()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes >= 0,
                    aux_targets - self._known_classes,
                    -1,
                )
                
                output = self._network(inputs, test=False)
                logits = output["logits"]

                loss = F.cross_entropy(logits, aux_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)

                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(aux_targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    epochs,
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.forward(inputs, test=True)["logits"]
            predicts = torch.max(outputs, dim=1)[1]          
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.backbone.multicount = 0
        self._network.backbone.zerocount = 0
        calc_task_acc = True
        
        if calc_task_acc:
            task_correct, task_acc, total = 0, 0, 0
            
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)

            with torch.no_grad():
                outputs = self._network.forward(inputs, test=True)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            
            # calculate the accuracy by using task_id
            if calc_task_acc:
                task_ids = (targets - self.init_cls) // self.inc + 1
                task_logits = torch.zeros(outputs.shape).to(self._device)
                for i, task_id in enumerate(task_ids):
                    if task_id == 0:
                        start_cls = 0
                        end_cls = self.init_cls
                    else:
                        start_cls = self.init_cls + (task_id-1)*self.inc
                        end_cls = self.init_cls + task_id*self.inc
                    task_logits[i, start_cls:end_cls] += outputs[i, start_cls:end_cls]
                # calculate the accuracy of task_id
                pred_task_ids = (torch.max(outputs, dim=1)[1] - self.init_cls) // self.inc + 1
                task_correct += (pred_task_ids.cpu() == task_ids).sum()
                
                pred_task_y = torch.max(task_logits, dim=1)[1]
                task_acc += (pred_task_y.cpu() == targets).sum()
                total += len(targets)

        if calc_task_acc:
            logging.info("Task correct: {}".format(tensor2numpy(task_correct) * 100 / total))
            logging.info("Task acc: {}".format(tensor2numpy(task_acc) * 100 / total))
        logging.info("multi-expert-num: {}".format(self._network.backbone.multicount))
        logging.info("zero-expert-num: {}".format(self._network.backbone.zerocount))
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]