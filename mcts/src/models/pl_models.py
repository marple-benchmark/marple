import time
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import lr_scheduler
import torch.nn as nn
import numpy as np
import src.simulator.vocab as vocab
from src.models.models import PolicyModel, TransformerHeadPolicyModel, HistAwareTransformerHeadPolicyModel, AudioConditionedTransformerHeadPolicyModel
from warmup_scheduler import GradualWarmupScheduler

from einops import rearrange, repeat, reduce


class BasicModel(pl.LightningModule):

    def __init__(self, args, model, model_name):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_cfg = args.optimizer
        self.batch_size = None
        self.model = model(vocab=vocab, args=args.model)
        self.configure_optimizers()
        self.model_name = model_name

    def forward(self, batch):
        gt_obj = batch['obj']
        gt_fur = batch['fur']
        gt_room = batch['room']
        gt_action = batch['action']
        if self.model_name == "policy":
            pred_obj, pred_fur, pred_room, pred_action = self.model(rearrange(batch['state'], 'b h w c -> b c h w'))
        elif self.model_name == "transformer":
            pred_obj, pred_fur, pred_room, pred_action = self.model(state=rearrange(batch['state'], 'b h w c -> b c h w'), obj=gt_obj, fur=gt_fur, room=gt_room, action=gt_action)
        elif self.model_name == "hist_aware_transformer":
            pred_obj, pred_fur, pred_room, pred_action = self.model(state=rearrange(batch['state'], 'b t h w c -> b t c h w'), obj=gt_obj, fur=gt_fur, room=gt_room, action=gt_action, state_mask=batch['state_mask'])
        elif self.model_name == "audio_transformer":
            pred_obj, pred_fur, pred_room, pred_action = self.model(state=rearrange(batch['state'], 'b h w c -> b c h w'), audio=batch["audios"], audio_mask=batch["audio_mask"], obj=gt_obj, fur=gt_fur, room=gt_room, action=gt_action)
        return gt_obj, gt_fur, gt_room, gt_action, pred_obj, pred_fur, pred_room, pred_action

    def compute_loss(self, gt_label, pred_label, prefix="train/"):
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(pred_label, gt_label)
        self.log(f"{prefix}_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        gt_obj, gt_fur, gt_room, gt_action, \
            pred_obj, pred_fur, pred_room, pred_action = self.forward(batch)
        obj_loss = self.compute_loss(gt_obj, pred_obj, prefix="loss/train_obj")
        fur_loss = self.compute_loss(gt_fur, pred_fur, prefix="loss/train_fur")
        room_loss = self.compute_loss(gt_room, pred_room, prefix="loss/train_room")
        action_loss = self.compute_loss(gt_action, pred_action, prefix="loss/train_action")
        loss = obj_loss + fur_loss + room_loss + action_loss
        self.log("loss/train", loss.item(), prog_bar=True, batch_size=self.batch_size)
        
        pred_obj = torch.argmax(pred_obj, dim=1)
        pred_fur = torch.argmax(pred_fur, dim=1)
        pred_room = torch.argmax(pred_room, dim=1)
        pred_action = torch.argmax(pred_action, dim=1)
        
        correct_objs = pred_obj == gt_obj
        correct_furs = pred_fur == gt_fur
        correct_rooms = pred_room == gt_room
        correct_actions = pred_action == gt_action
        
        correct = correct_objs & correct_furs & correct_rooms & correct_actions
        num = len(correct) * 1.0 

        self.log('accuracy/train', torch.sum(correct).item() / num)
        self.log('accuracy/train_obj', torch.sum(correct_objs).item() / num)
        self.log('accuracy/train_fur', torch.sum(correct_furs).item() / num)
        self.log('accuracy/train_room', torch.sum(correct_rooms).item() / num)
        self.log('accuracy/train_action', torch.sum(correct_actions).item() / num) 
        
        return loss

    def validation_step(self, batch, batch_idx):
        gt_obj, gt_fur, gt_room, gt_action, \
            pred_obj, pred_fur, pred_room, pred_action = self.forward(batch)

        obj_loss = self.compute_loss(gt_obj, pred_obj, prefix="loss/val_obj")
        fur_loss = self.compute_loss(gt_fur, pred_fur, prefix="loss/val_fur")
        room_loss = self.compute_loss(gt_room, pred_room, prefix="loss/val_room")
        action_loss = self.compute_loss(gt_action, pred_action, prefix="loss/val_action")
        loss = obj_loss + fur_loss + room_loss + action_loss
        self.log("loss/val", loss.item(), prog_bar=True, batch_size=self.batch_size)

        pred_obj = torch.argmax(pred_obj, dim=1)
        pred_fur = torch.argmax(pred_fur, dim=1)
        pred_room = torch.argmax(pred_room, dim=1)
        pred_action = torch.argmax(pred_action, dim=1)
        
        correct_objs = pred_obj == gt_obj
        correct_furs = pred_fur == gt_fur
        correct_rooms = pred_room == gt_room
        correct_actions = pred_action == gt_action
        
        correct = correct_objs & correct_furs & correct_rooms & correct_actions
        num = len(correct) * 1.0 

        self.log('accuracy/val', torch.sum(correct).item() / num)
        self.log('accuracy/val_obj', torch.sum(correct_objs).item() / num)
        self.log('accuracy/val_fur', torch.sum(correct_furs).item() / num)
        self.log('accuracy/val_room', torch.sum(correct_rooms).item() / num)
        self.log('accuracy/val_action', torch.sum(correct_actions).item() / num) 
        
    def test_step(self, batch, batch_idx):
        metrics = {}
        gt_obj, gt_fur, gt_room, gt_action, pred_obj, pred_fur, pred_room, pred_action = self.forward(batch)

        obj_loss = self.compute_loss(gt_obj, pred_obj, prefix="loss/test_obj")
        fur_loss = self.compute_loss(gt_fur, pred_fur, prefix="loss/test_fur")
        room_loss = self.compute_loss(gt_room, pred_room, prefix="loss/test_room")
        action_loss = self.compute_loss(gt_action, pred_action, prefix="loss/test_action")
        
        metrics["loss/test"] = obj_loss + fur_loss + room_loss + action_loss 

        pred_obj = torch.argmax(pred_obj, dim=1)
        pred_fur = torch.argmax(pred_fur, dim=1)
        pred_room = torch.argmax(pred_room, dim=1)
        pred_action = torch.argmax(pred_action, dim=1)
        
        correct_objs = pred_obj == gt_obj
        correct_furs = pred_fur == gt_fur
        correct_rooms = pred_room == gt_room
        correct_actions = pred_action == gt_action
        

        correct = correct_objs & correct_furs & correct_rooms & correct_actions
        num = len(correct) * 1.0
        metrics['accuracy/test'] = torch.sum(correct).item() / num
        metrics['accuracy/test_obj'] = torch.sum(correct_objs).item() / num
        metrics['accuracy/test_fur'] = torch.sum(correct_furs).item() / num
        metrics['accuracy/test_room'] = torch.sum(correct_rooms).item() / num
        metrics['accuracy/test_action'] = torch.sum(correct_actions).item() / num

        self.log_dict(metrics)

        return metrics

    def configure_optimizers(self):
        weight_decay = self.optimizer_cfg.weight_decay if "weight_decay" in self.optimizer_cfg else 0
        use_lr_scheduler = self.optimizer_cfg.use_lr_scheduler if "use_lr_scheduler" in self.optimizer_cfg else False
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr, weight_decay=weight_decay) # 1e-5
        if use_lr_scheduler:
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.optimizer_cfg.lr_restart)
            scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=self.optimizer_cfg.warmup, after_scheduler=scheduler)
            return [optimizer], [{"scheduler": scheduler_warmup, "interval": "epoch"}]
        else:
            return optimizer
                
    def inference(self, state, state_mask=None):

        B = 1

        if self.model_name == "transformer" or self.model_name == "hist_aware_transformer":

            input_obj = torch.zeros((B), dtype=torch.long).to(self.device)
            input_fur = torch.zeros((B), dtype=torch.long).to(self.device)
            input_room = torch.zeros((B), dtype=torch.long).to(self.device)
            input_action = torch.zeros((B), dtype=torch.long).to(self.device)

            if self.model_name == "hist_aware_transformer":
                assert state_mask is not None, "state mask must be provided for hist aware transformer"
                state = torch.tensor(state, dtype=torch.float32).to(self.device)  # T, H, W, C
                state = repeat(state, 't h w c -> b t c h w', b=B)
                state_mask = torch.tensor(state_mask, dtype=torch.int32).to(self.device)  # T
                state_mask = repeat(state_mask, 't -> b t', b=B)
            else:
                state = torch.tensor(state, dtype=torch.float32).to(self.device)  # H, W, C
                state = repeat(state, 'h w c -> b c h w', b=B)
                assert state_mask is None, "state mask must not be provided for normal transformer"

            with torch.no_grad():
                # Note that the order transformer predicts is: action, room, fur, obj
                _, _, _, pred_action_logit = self.model(state, input_obj, input_fur, input_room, input_action, state_mask=state_mask)  # B, V
                pred_action = torch.argmax(pred_action_logit, dim=1) # B, 1

                _, _, pred_room_logit, _ = self.model(state, input_obj, input_fur, input_room, pred_action, state_mask=state_mask)  # B, V
                pred_room = torch.argmax(pred_room_logit, dim=1)  # B, 1

                _, pred_fur_logit, _, _ = self.model(state, input_obj, input_fur, pred_room, pred_action, state_mask=state_mask)  # B, V
                pred_fur = torch.argmax(pred_fur_logit, dim=1) # B, 1

                pred_obj_logit, _, _, _ = self.model(state, input_obj, pred_fur, pred_room, pred_action, state_mask=state_mask)  # B, V
                pred_obj = torch.argmax(pred_obj_logit, dim=1)  # B, 1
        else:
            raise NotImplementedError

        # TODO: implement other sampling methods

        return pred_obj, pred_fur, pred_room, pred_action

    def inference_sample(self, state, state_mask=None, temp=1.):
        device = 'cuda' if torch.cuda.is_available else 'cpu'
        self.to(device)
        state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)  # H, W, C
        state = torch.squeeze(state)

        # if self.model_name == "hist_aware_transformer":
        #     assert state_mask is not None, "state mask must be provided for hist aware transformer"
        #     state = torch.tensor(state, dtype=torch.float32).to(self.device)  # T, H, W, C
        #     state = repeat(state, 't h w c -> b t c h w', b=B)
        #     state_mask = torch.tensor(state_mask, dtype=torch.int32).to(self.device)  # T
        #     state_mask = repeat(state_mask, 't -> b t', b=B)
        # else:

        if len(state.shape) == 3:
            state = repeat(state, 'h w c -> b c h w', b=1)
        else:
            state = repeat(state, 'b h w c -> b c h w')#, b=1)

        assert state_mask is None, "state mask must not be provided for normal transformer"

        B = state.shape[0]

        input_obj = torch.zeros((B), dtype=torch.long).to(self.device)
        input_fur = torch.zeros((B), dtype=torch.long).to(self.device)
        input_room = torch.zeros((B), dtype=torch.long).to(self.device)
        input_action = torch.zeros((B), dtype=torch.long).to(self.device)

        with torch.no_grad():
            _, _, _, pred_action_logit = self.model(state, input_obj, input_fur, input_room, input_action, state_mask=state_mask)  # B, V
            pred_action_logit = torch.softmax(pred_action_logit / temp, 1) 

            pred_action = torch.multinomial(pred_action_logit, 1).squeeze()

            _, _, pred_room_logit, _ = self.model(state, input_obj, input_fur, input_room, pred_action, state_mask=state_mask)  # B, V
            pred_room_logit = torch.softmax(pred_room_logit / temp, 1)          
            pred_room = torch.multinomial(pred_room_logit, 1).squeeze()
            _, pred_fur_logit, _, _ = self.model(state, input_obj, input_fur, pred_room, pred_action, state_mask=state_mask)  # B, V
            pred_fur_logit = torch.softmax(pred_fur_logit / temp, 1) 
            pred_fur = torch.multinomial(pred_fur_logit, 1).squeeze()
            pred_obj_logit, _, _, _ = self.model(state, input_obj, pred_fur, pred_room, pred_action, state_mask=state_mask)  # B, V
            pred_obj_logit = torch.softmax(pred_obj_logit / temp, 1) 
            pred_obj = torch.multinomial(pred_obj_logit, 1).squeeze()
            
        self.to('cpu')
        return pred_obj.cpu(), pred_fur.cpu(), pred_room.cpu(), pred_action.cpu()

    def sample_actions(self, state, vocab, decode_strategy="beam", num_samples=20, debug=False, state_mask=None):
        """

        :param state: features of dim [h, w, c] if transformer, [t, h, w, c] if hist aware transformer
        :param state_mask: mask of dim [t] if hist aware transformer, ow None
        :param vocab: used to automatically adjust beam size if vocab size is smaller than beam size
        :param decode_strategy: options are beam, random
        :param num_samples: for random, controls the number of random samples. For beam, controls the beam size.
        :return: each sample consists of a tuple of (obj, fur, room, action, prob)
            - pred_obj: num_samples
            - pred_fur: num_samples
            - pred_room: num_samples
            - pred_action: num_samples
            - pred_prob: num_samples
        """

        # this function should take in random number of inputs
        def print_if_debug(str):
            if debug:
                print(str)

        assert decode_strategy in ["beam", "random"]

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if state_mask is not None:
            state_mask = torch.tensor(state_mask, dtype=torch.int32).to(self.device)
        # need to fix this here; are we accumulating 

        if self.model_name == "hist_aware_transformer":
            assert state_mask is not None, "state mask must be provided for hist aware transformer"
            state_mask = torch.tensor(state_mask, dtype=torch.int32).to(self.device)
            state = repeat(state, 't h w c -> b t c h w', b=num_samples)
            state_mask = repeat(state_mask, 't -> b t', b=num_samples)
        elif self.model_name == "transformer":
            assert state_mask is None, "state mask must not be provided for normal transformer"
            state = repeat(state, 'h w c -> b c h w', b=num_samples)
        else:
            raise NotImplementedError

        if self.model_name == "transformer" or self.model_name == "hist_aware_transformer":
            input_obj = torch.zeros((num_samples), dtype=torch.long).to(self.device)
            input_fur = torch.zeros((num_samples), dtype=torch.long).to(self.device)
            input_room = torch.zeros((num_samples), dtype=torch.long).to(self.device)
            input_action = torch.zeros((num_samples), dtype=torch.long).to(self.device)

            # random decoding
            if decode_strategy == "random":
                # Note that the order transformer predicts is: action, room, fur, obj
                with torch.no_grad():
                    _, _, _, pred_action_logit = self.model(state, input_obj, input_fur, input_room, input_action, state_mask=state_mask)  # B, V
                pred_action_prob = F.softmax(pred_action_logit, dim=1)  # B, V

                sampled_action = torch.multinomial(pred_action_prob, 1).squeeze()  # B
                sampled_action_prob = pred_action_prob[torch.arange(num_samples), sampled_action]  # B
                print_if_debug(f"sampled_action_prob ({sampled_action_prob.shape}): {sampled_action_prob}")
                print_if_debug(f"sampled_action ({sampled_action.shape}): {sampled_action}")

                with torch.no_grad():
                    _, _, pred_room_logit, _ = self.model(state, input_obj, input_fur, input_room, sampled_action, state_mask=state_mask)  # B, V
                pred_room_prob = F.softmax(pred_room_logit, dim=1)  # B, V

                sampled_room = torch.multinomial(pred_room_prob, 1).squeeze()  # B
                sampled_room_prob = pred_room_prob[torch.arange(num_samples), sampled_room]  # B
                print_if_debug(f"sampled_room_prob ({sampled_room_prob.shape}): {sampled_room_prob}")
                print_if_debug(f"sampled_room ({sampled_room.shape}): {sampled_room}")

                with torch.no_grad():
                    _, pred_fur_logit, _, _ = self.model(state, input_obj, input_fur, sampled_room, sampled_action, state_mask=state_mask)  # B, V
                pred_fur_prob = F.softmax(pred_fur_logit, dim=1)  # B, V                                                                            5

                sampled_fur = torch.multinomial(pred_fur_prob, 1).squeeze()  # B
                sampled_fur_prob = pred_fur_prob[torch.arange(num_samples), sampled_fur]  # B
                print_if_debug(f"sampled_fur_prob ({sampled_fur_prob.shape}): {sampled_fur_prob}")
                print_if_debug(f"sampled_fur ({sampled_fur.shape}): {sampled_fur}")

                with torch.no_grad():
                    pred_obj_logit, _, _, _ = self.model(state, input_obj, sampled_fur, sampled_room, sampled_action, state_mask=state_mask)  # B, V
                pred_obj_prob = F.softmax(pred_obj_logit, dim=1)  # B, V

                sampled_obj = torch.multinomial(pred_obj_prob, 1).squeeze()  # B
                sampled_obj_prob = pred_obj_prob[torch.arange(num_samples), sampled_obj]  # B
                print_if_debug(f"sampled_obj_prob ({sampled_obj_prob.shape}): {sampled_obj_prob}")
                print_if_debug(f"sampled_obj ({sampled_obj.shape}): {sampled_obj}")

                # compute joint probabilities
                sampled_joint_prob = sampled_action_prob * sampled_room_prob * sampled_fur_prob * sampled_obj_prob
                print_if_debug(f"sampled_joint_prob ({sampled_joint_prob.shape}): {sampled_joint_prob}")

                # sort by joint probabilities
                sorted_joint_prob, sorted_joint_prob_idx = torch.sort(sampled_joint_prob, descending=True)
                print("-----------------------------\nprediction:\naction room furniture object: prob")
                for jpi in sorted_joint_prob_idx:
                    print(f"{sampled_action[jpi]} {sampled_room[jpi]} {sampled_fur[jpi]} {sampled_obj[jpi]}: {sampled_joint_prob[jpi]}")
                print("-----------------------------")

                return sampled_obj, sampled_fur, sampled_room, sampled_action, sampled_joint_prob

            elif decode_strategy == "beam":
                # beam search decoding
                # Note that the order transformer predicts is: action, room, fur, obj

                # 1. find top k action
                with torch.no_grad():
                    _, _, _, pred_a_logit = self.model(state, input_obj, input_fur, input_room, input_action, state_mask=state_mask)  # B, V
                pred_a_prob = F.softmax(pred_a_logit, dim=1)  # B, V

                k_a = min(num_samples, len(vocab.SUBGOAL_ACTION_TO_IDX))
                print_if_debug(f"k action: {k_a}")
                sampled_a_prob, sampled_a = torch.topk(pred_a_prob[0], k=k_a)  # k_a
                print_if_debug(f"sampled_a_prob ({sampled_a_prob.shape}): {sampled_a_prob}")
                print_if_debug(f"sampled_a ({sampled_a.shape}): {sampled_a}")

                # 2 find top k (action, room)
                with torch.no_grad():
                    _, _, pred_r_logit, _ = self.model(state[:k_a], input_obj[:k_a], input_fur[:k_a], input_room[:k_a], sampled_a, state_mask=state_mask[:k_a] if state_mask is not None else None)  # k_action, V
                pred_r_prob = F.softmax(pred_r_logit, dim=1)  # k_action, V
                print_if_debug(f"pred_r_prob ({pred_r_prob.shape}): {pred_r_prob}")

                # compute joint probabilities
                pred_ar_prob = repeat(sampled_a_prob, 'k -> k v', v=len(vocab.ROOM_TO_IDX)) * pred_r_prob  # k, v_room
                print_if_debug(f"pred_ar_prob ({pred_ar_prob.shape}): {pred_ar_prob}")

                k_ar = min(num_samples, len(vocab.SUBGOAL_ACTION_TO_IDX) * len(vocab.ROOM_TO_IDX))
                print_if_debug(f"k action-room: {k_ar}")
                sampled_ar_prob, sampled_ar = torch.topk(pred_ar_prob.view(-1), k=k_ar)  # k_ar

                sampled_a_idx = sampled_ar // len(vocab.ROOM_TO_IDX)  # k
                sampled_r = sampled_ar % len(vocab.ROOM_TO_IDX)  # k
                sampled_a = sampled_a[sampled_a_idx]  # k

                print_if_debug(f"sampled_ar_prob ({sampled_ar_prob.shape}): {sampled_ar_prob}")
                print_if_debug(f"sampled_a_idx ({sampled_a_idx.shape}): {sampled_a_idx}")
                print_if_debug(f"sampled_a ({sampled_a.shape}): {sampled_a}")
                print_if_debug(f"sampled_r ({sampled_r.shape}): {sampled_r}")

                # 3 find top k (action, room, fur)
                with torch.no_grad():
                    _, pred_f_logit, _, _ = self.model(state[:k_ar], input_obj[:k_ar], input_fur[:k_ar], sampled_r, sampled_a, state_mask=state_mask[:k_ar] if state_mask is not None else None)
                pred_f_prob = F.softmax(pred_f_logit, dim=1)  # k_ar, V
                print_if_debug(f"pred_f_prob ({pred_f_prob.shape}): {pred_f_prob}")

                # compute joint probabilities
                # note that furniture uses the object vocab
                pred_arf_prob = repeat(sampled_ar_prob, 'k -> k v', v=len(vocab.OBJECT_TO_IDX)) * pred_f_prob  # k, v_fur
                print_if_debug(f"pred_arf_prob ({pred_arf_prob.shape}): {pred_arf_prob}")

                k_arf = min(num_samples, len(vocab.SUBGOAL_ACTION_TO_IDX) * len(vocab.ROOM_TO_IDX) * len(vocab.OBJECT_TO_IDX))
                print_if_debug(f"k action-room-fur: {k_arf}")
                sampled_arf_prob, sampled_arf = torch.topk(pred_arf_prob.view(-1), k=k_arf)  # k_arf

                sampled_ar_idx = sampled_arf // len(vocab.OBJECT_TO_IDX)
                sampled_f = sampled_arf % len(vocab.OBJECT_TO_IDX)
                sampled_r = sampled_r[sampled_ar_idx]
                sampled_a = sampled_a[sampled_ar_idx]

                print_if_debug(f"sampled_arf_prob ({sampled_arf_prob.shape}): {sampled_arf_prob}")
                print_if_debug(f"sampled_ar_idx ({sampled_ar_idx.shape}): {sampled_ar_idx}")
                print_if_debug(f"sampled_a ({sampled_a.shape}): {sampled_a}")
                print_if_debug(f"sampled_r ({sampled_r.shape}): {sampled_r}")
                print_if_debug(f"sampled_f ({sampled_f.shape}): {sampled_f}")

                # 4 find top k (action, room, fur, obj)
                with torch.no_grad():
                    pred_o_logit, _, _, _ = self.model(state[:k_arf], input_obj[:k_arf], sampled_f, sampled_r, sampled_a, state_mask=state_mask[:k_arf] if state_mask is not None else None)
                pred_o_prob = F.softmax(pred_o_logit, dim=1)  # k_arf, V
                print_if_debug(f"pred_o_prob ({pred_o_prob.shape}): {pred_o_prob}")

                # compute joint probabilities
                pred_arfo_prob = repeat(sampled_arf_prob, 'k -> k v', v=len(vocab.OBJECT_TO_IDX)) * pred_o_prob  # k, v_obj
                print_if_debug(f"pred_arfo_prob ({pred_arfo_prob.shape}): {pred_arfo_prob}")

                k_arfo = min(num_samples, len(vocab.SUBGOAL_ACTION_TO_IDX) * len(vocab.ROOM_TO_IDX) * len(vocab.FURNITURE_TO_IDX) * len(vocab.OBJECT_TO_IDX))
                print_if_debug(f"k action-room-fur-obj: {k_arfo}")
                sampled_arfo_prob, sampled_arfo = torch.topk(pred_arfo_prob.view(-1), k=k_arfo)  # k_arfo

                sampled_arf_idx = sampled_arfo // len(vocab.OBJECT_TO_IDX)
                sampled_o = sampled_arfo % len(vocab.OBJECT_TO_IDX)
                sampled_f = sampled_f[sampled_arf_idx]
                sampled_r = sampled_r[sampled_arf_idx]
                sampled_a = sampled_a[sampled_arf_idx]

                print_if_debug(f"sampled_arfo_prob ({sampled_arfo_prob.shape}): {sampled_arfo_prob}")
                print_if_debug(f"sampled_arf_idx ({sampled_arf_idx.shape}): {sampled_arf_idx}")
                print_if_debug(f"sampled_a ({sampled_a.shape}): {sampled_a}")
                print_if_debug(f"sampled_r ({sampled_r.shape}): {sampled_r}")
                print_if_debug(f"sampled_f ({sampled_f.shape}): {sampled_f}")
                print_if_debug(f"sampled_o ({sampled_o.shape}): {sampled_o}")

                print("-----------------------------\nprediction:\naction room furniture object: prob")
                for jpi, prob in enumerate(sampled_arfo_prob):
                    print(f"{sampled_a[jpi]} {sampled_r[jpi]} {sampled_f[jpi]} {sampled_o[jpi]}: {prob}")
                print("-----------------------------")

                return sampled_o, sampled_f, sampled_r, sampled_a, sampled_arfo_prob

            else:
                raise KeyError(f"Invalid decode strategy: {decode_strategy}")

    def sample_actions_batch(self, state, vocab, decode_strategy="beam", num_samples=20, debug=False, state_mask=None):
        """

        :param state: features of dim [b, h, w, c] if transformer, [b, t, h, w, c] if hist aware transformer
        :param state_mask: mask of dim [b t] if hist aware transformer, ow None
        :param vocab: used to automatically adjust beam size if vocab size is smaller than beam size
        :param decode_strategy: options are beam, random
        :param num_samples: for random, controls the number of random samples. For beam, controls the beam size.
        :return: each sample consists of a tuple of (obj, fur, room, action, prob)
            - pred_obj: num_samples
            - pred_fur: num_samples
            - pred_room: num_samples
            - pred_action: num_samples
            - pred_prob: num_samples
        """

        # this function should take in random number of inputs
        def print_if_debug(str):
            if debug:
                print(str)

        def _c(tensor):
            # collapse
            # input: b, k, ...
            # return b*k, ...
            return tensor.reshape(-1, *tensor.shape[2:])

        def _e(tensor, b):
            # expand
            # input: b*k, ...
            # return b, k, ...
            return tensor.reshape(b, -1, *tensor.shape[1:])

        def _ekc(tensor, ki, b):
            tensor = _e(tensor, b) # b, k, ...
            tensor = tensor[:, :ki] # b, ki, ...
            tensor = _c(tensor) # b*ki, ...
            return tensor


        assert decode_strategy in ["beam", "random"]

        B = state.shape[0]

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        if state_mask is not None:
            state_mask = torch.tensor(state_mask, dtype=torch.int32).to(self.device)

        if self.model_name == "hist_aware_transformer":
            assert state_mask is not None, "state mask must be provided for hist aware transformer"
            state = repeat(state, 'b t h w c -> (b k) t c h w', k=num_samples)
            state_mask = repeat(state_mask, 'b t -> (b k) t', k=num_samples)
        elif self.model_name == "transformer":
            assert state_mask is None, "state mask must not be provided for normal transformer"
            state = repeat(state, 'b h w c -> (b k) c h w', k=num_samples)
        else:
            raise NotImplementedError

        if self.model_name == "transformer" or self.model_name == "hist_aware_transformer":
            input_obj = torch.zeros((B * num_samples), dtype=torch.long).to(self.device)
            input_fur = torch.zeros((B * num_samples), dtype=torch.long).to(self.device)
            input_room = torch.zeros((B * num_samples), dtype=torch.long).to(self.device)
            input_action = torch.zeros((B * num_samples), dtype=torch.long).to(self.device)

            # random decoding
            if decode_strategy == "random":
                # Note that the order transformer predicts is: action, room, fur, obj
                # Note that self.model() takes about 7 seconds for BK = 400
                with torch.no_grad():
                    _, _, _, pred_action_logit = self.model(state, input_obj, input_fur, input_room, input_action, state_mask=state_mask)  # BK, V
                pred_action_prob = F.softmax(pred_action_logit, dim=1)  # BK, V

                sampled_action = torch.multinomial(pred_action_prob, 1).squeeze()  # BK
                sampled_action_prob = pred_action_prob[torch.arange(B * num_samples), sampled_action]  # BK
                print_if_debug(f"sampled_action_prob ({sampled_action_prob.shape}): {sampled_action_prob}")
                print_if_debug(f"sampled_action ({sampled_action.shape}): {sampled_action}")

                with torch.no_grad():
                    _, _, pred_room_logit, _ = self.model(state, input_obj, input_fur, input_room, sampled_action, state_mask=state_mask)  # BK, V
                pred_room_prob = F.softmax(pred_room_logit, dim=1)  # BK, V

                sampled_room = torch.multinomial(pred_room_prob, 1).squeeze()  # BK
                sampled_room_prob = pred_room_prob[torch.arange(B * num_samples), sampled_room]  # BK
                print_if_debug(f"sampled_room_prob ({sampled_room_prob.shape}): {sampled_room_prob}")
                print_if_debug(f"sampled_room ({sampled_room.shape}): {sampled_room}")

                with torch.no_grad():
                    _, pred_fur_logit, _, _ = self.model(state, input_obj, input_fur, sampled_room, sampled_action, state_mask=state_mask)  # BK, V
                pred_fur_prob = F.softmax(pred_fur_logit, dim=1)  # BK, V

                sampled_fur = torch.multinomial(pred_fur_prob, 1).squeeze()  # BK
                sampled_fur_prob = pred_fur_prob[torch.arange(B * num_samples), sampled_fur]  # BK
                print_if_debug(f"sampled_fur_prob ({sampled_fur_prob.shape}): {sampled_fur_prob}")
                print_if_debug(f"sampled_fur ({sampled_fur.shape}): {sampled_fur}")

                with torch.no_grad():
                    pred_obj_logit, _, _, _ = self.model(state, input_obj, sampled_fur, sampled_room, sampled_action, state_mask=state_mask)  # BK, V
                pred_obj_prob = F.softmax(pred_obj_logit, dim=1)  # BK, V

                sampled_obj = torch.multinomial(pred_obj_prob, 1).squeeze()  # BK
                sampled_obj_prob = pred_obj_prob[torch.arange(B * num_samples), sampled_obj]  # BK
                print_if_debug(f"sampled_obj_prob ({sampled_obj_prob.shape}): {sampled_obj_prob}")
                print_if_debug(f"sampled_obj ({sampled_obj.shape}): {sampled_obj}")

                # compute joint probabilities
                sampled_joint_prob = sampled_action_prob * sampled_room_prob * sampled_fur_prob * sampled_obj_prob
                print_if_debug(f"sampled_joint_prob ({sampled_joint_prob.shape}): {sampled_joint_prob}")

                sampled_joint_prob = _e(sampled_joint_prob, B)  # B, K
                sampled_action = _e(sampled_action, B)  # B, K
                sampled_room = _e(sampled_room, B)  # B, K
                sampled_fur = _e(sampled_fur, B)  # B, K
                sampled_obj = _e(sampled_obj, B)  # B, K

                # sort by joint probabilities
                sorted_joint_prob, sorted_joint_prob_idx = torch.sort(sampled_joint_prob, descending=True, dim=-1) # B, K
                print("=============================\nprediction:\naction room furniture object: prob")
                for bi in range(B):
                    print("-----------------------------\nbatch {}".format(bi))
                    for jpi in sorted_joint_prob_idx[bi]:
                        print(f"{sampled_action[bi][jpi]} {sampled_room[bi][jpi]} {sampled_fur[bi][jpi]} {sampled_obj[bi][jpi]}: {sampled_joint_prob[bi][jpi]}")
                print("=============================")

                return sampled_obj, sampled_fur, sampled_room, sampled_action, sampled_joint_prob

            elif decode_strategy == "beam":
                # beam search decoding
                # Note that the order transformer predicts is: action, room, fur, obj

                # 1. find top k action
                with torch.no_grad():
                    _, _, _, pred_a_logit = self.model(state, input_obj, input_fur, input_room, input_action, state_mask=state_mask)  # BK, V
                pred_a_prob = F.softmax(pred_a_logit, dim=1)  # BK, V
                pred_a_prob = _e(pred_a_prob, B)  # B, K, V

                k_a = min(num_samples, len(vocab.SUBGOAL_ACTION_TO_IDX))
                print_if_debug(f"k action: {k_a}")
                sampled_a_prob, sampled_a = torch.topk(pred_a_prob[:, 0], k=k_a, dim=-1)  # B, k_a
                print_if_debug(f"sampled_a_prob ({sampled_a_prob.shape}): {sampled_a_prob}")
                print_if_debug(f"sampled_a ({sampled_a.shape}): {sampled_a}")

                # 2 find top k (action, room)
                with torch.no_grad():
                    if self.model_name == "transformer":
                        _, _, pred_r_logit, _ = self.model(_ekc(state, k_a, B),
                                                        _ekc(input_obj, k_a, B),
                                                        _ekc(input_fur, k_a, B),
                                                        _ekc(input_room, k_a, B),
                                                        _c(sampled_a),
                                                        state_mask=_ekc(state_mask, k_a, B) if state_mask else None)  # B*k_action, V
                    elif self.model_name == "hist_aware_transformer":
                        _, _, pred_r_logit, _ = self.model(_ekc(state, k_a, B),
                                                        _ekc(input_obj, k_a, B),
                                                        _ekc(input_fur, k_a, B),
                                                        _ekc(input_room, k_a, B),
                                                        _c(sampled_a),
                                                        state_mask=_ekc(state_mask, k_a, B) if state_mask.any() else None)
                pred_r_prob = F.softmax(pred_r_logit, dim=1)  # B*k_action, V
                pred_r_prob = _e(pred_r_prob, B)  # B, k_action, V
                print_if_debug(f"pred_r_prob ({pred_r_prob.shape}): {pred_r_prob}")

                # compute joint probabilities
                pred_ar_prob = repeat(sampled_a_prob, 'b k -> b k v', v=len(vocab.ROOM_TO_IDX)) * pred_r_prob  # B, k_action, v_room
                print_if_debug(f"pred_ar_prob ({pred_ar_prob.shape}): {pred_ar_prob}")

                pred_ar_prob = rearrange(pred_ar_prob, 'b k v -> b (k v)')  # B, k_action * v_room

                k_ar = min(num_samples, len(vocab.SUBGOAL_ACTION_TO_IDX) * len(vocab.ROOM_TO_IDX))
                print_if_debug(f"k action-room: {k_ar}")
                sampled_ar_prob, sampled_ar = torch.topk(pred_ar_prob, k=k_ar, dim=-1)  # B, k_ar

                sampled_a_idx = sampled_ar // len(vocab.ROOM_TO_IDX)  # B, k_ar
                sampled_r = sampled_ar % len(vocab.ROOM_TO_IDX)  # B, k_ar
                sampled_a = _c(sampled_a)[sampled_a_idx]  # B, k_ar

                print_if_debug(f"sampled_ar_prob ({sampled_ar_prob.shape}): {sampled_ar_prob}")
                print_if_debug(f"sampled_a_idx ({sampled_a_idx.shape}): {sampled_a_idx}")
                print_if_debug(f"sampled_a ({sampled_a.shape}): {sampled_a}")
                print_if_debug(f"sampled_r ({sampled_r.shape}): {sampled_r}")

                # 3 find top k (action, room, fur)
                with torch.no_grad():
                    if self.model_name == "transformer":
                        _, pred_f_logit, _, _ = self.model(_ekc(state, k_ar, B),
                                                        _ekc(input_obj, k_ar, B),
                                                        _ekc(input_fur, k_ar, B),
                                                        _c(sampled_r),
                                                        _c(sampled_a),
                                                        state_mask=_ekc(state_mask, k_ar, B) if state_mask else None)  # B*k_ar, V
                    elif self.model_name == "hist_aware_transformer":
                        _, pred_f_logit, _, _ = self.model(_ekc(state, k_ar, B),
                                                        _ekc(input_obj, k_ar, B),
                                                        _ekc(input_fur, k_ar, B),
                                                        _c(sampled_r),
                                                        _c(sampled_a),
                                                        state_mask=_ekc(state_mask, k_ar, B) if state_mask.any() else None)

                pred_f_prob = F.softmax(pred_f_logit, dim=1)  # B*k_ar, V
                pred_f_prob = _e(pred_f_prob, B)  # B, k_ar, V
                print_if_debug(f"pred_f_prob ({pred_f_prob.shape}): {pred_f_prob}")

                # compute joint probabilities
                # note that furniture uses the object vocab
                pred_arf_prob = repeat(sampled_ar_prob, 'b k -> b k v', v=len(vocab.OBJECT_TO_IDX)) * pred_f_prob  # B, k, v_fur
                print_if_debug(f"pred_arf_prob ({pred_arf_prob.shape}): {pred_arf_prob}")

                pred_arf_prob = rearrange(pred_arf_prob, 'b k v -> b (k v)')  # B, k_ar * v_fur

                k_arf = min(num_samples, len(vocab.SUBGOAL_ACTION_TO_IDX) * len(vocab.ROOM_TO_IDX) * len(vocab.OBJECT_TO_IDX))
                print_if_debug(f"k action-room-fur: {k_arf}")
                sampled_arf_prob, sampled_arf = torch.topk(pred_arf_prob, k=k_arf)  # B, k_arf

                sampled_ar_idx = sampled_arf // len(vocab.OBJECT_TO_IDX)
                sampled_f = sampled_arf % len(vocab.OBJECT_TO_IDX)
                sampled_r = _c(sampled_r)[sampled_ar_idx]
                sampled_a = _c(sampled_a)[sampled_ar_idx]

                print_if_debug(f"sampled_arf_prob ({sampled_arf_prob.shape}): {sampled_arf_prob}")
                print_if_debug(f"sampled_ar_idx ({sampled_ar_idx.shape}): {sampled_ar_idx}")
                print_if_debug(f"sampled_a ({sampled_a.shape}): {sampled_a}")
                print_if_debug(f"sampled_r ({sampled_r.shape}): {sampled_r}")
                print_if_debug(f"sampled_f ({sampled_f.shape}): {sampled_f}")

                # 4 find top k (action, room, fur, obj)
                with torch.no_grad():
                    if self.model_name == "transformer":
                        pred_o_logit, _, _, _ = self.model(_ekc(state, k_arf, B),
                                                        _ekc(input_obj, k_arf, B),
                                                        _c(sampled_f),
                                                        _c(sampled_r),
                                                        _c(sampled_a),
                                                        state_mask=_ekc(state_mask, k_arf, B) if state_mask else None)  # B*k_arf, V
                    elif self.model_name == "hist_aware_transformer":
                        pred_o_logit, _, _, _ = self.model(_ekc(state, k_arf, B),
                                                        _ekc(input_obj, k_arf, B),
                                                        _c(sampled_f),
                                                        _c(sampled_r),
                                                        _c(sampled_a),
                                                        state_mask=_ekc(state_mask, k_arf, B) if state_mask.any() else None)  # B*k_arf, V
                pred_o_prob = F.softmax(pred_o_logit, dim=1)  # B * k_arf, V
                pred_o_prob = _e(pred_o_prob, B)  # B, k_arf, V
                print_if_debug(f"pred_o_prob ({pred_o_prob.shape}): {pred_o_prob}")

                # compute joint probabilities
                pred_arfo_prob = repeat(sampled_arf_prob, 'b k -> b k v', v=len(vocab.OBJECT_TO_IDX)) * pred_o_prob  # B, k, v_obj
                print_if_debug(f"pred_arfo_prob ({pred_arfo_prob.shape}): {pred_arfo_prob}")

                pred_arfo_prob = rearrange(pred_arfo_prob, 'b k v -> b (k v)')  # B, k_arf * v_obj

                k_arfo = min(num_samples, len(vocab.SUBGOAL_ACTION_TO_IDX) * len(vocab.ROOM_TO_IDX) * len(vocab.FURNITURE_TO_IDX) * len(vocab.OBJECT_TO_IDX))
                print_if_debug(f"k action-room-fur-obj: {k_arfo}")
                sampled_arfo_prob, sampled_arfo = torch.topk(pred_arfo_prob, k=k_arfo)  # B, k_arfo

                sampled_arf_idx = sampled_arfo // len(vocab.OBJECT_TO_IDX)
                sampled_o = sampled_arfo % len(vocab.OBJECT_TO_IDX)
                sampled_f = _c(sampled_f)[sampled_arf_idx]
                sampled_r = _c(sampled_r)[sampled_arf_idx]
                sampled_a = _c(sampled_a)[sampled_arf_idx]

                print_if_debug(f"sampled_arfo_prob ({sampled_arfo_prob.shape}): {sampled_arfo_prob}")
                print_if_debug(f"sampled_arf_idx ({sampled_arf_idx.shape}): {sampled_arf_idx}")
                print_if_debug(f"sampled_a ({sampled_a.shape}): {sampled_a}")
                print_if_debug(f"sampled_r ({sampled_r.shape}): {sampled_r}")
                print_if_debug(f"sampled_f ({sampled_f.shape}): {sampled_f}")
                print_if_debug(f"sampled_o ({sampled_o.shape}): {sampled_o}")

                print("=============================\nprediction:\naction room furniture object: prob")
                for bi in range(B):
                    print("-----------------------------\nbatch {}".format(bi))
                    for jpi, prob in enumerate(sampled_arfo_prob[bi]):
                        print(f"{sampled_a[bi][jpi]} {sampled_r[bi][jpi]} {sampled_f[bi][jpi]} {sampled_o[bi][jpi]}: {prob}")
                print("=============================")

                return sampled_o, sampled_f, sampled_r, sampled_a, sampled_arfo_prob

            else:
                raise KeyError(f"Invalid decode strategy: {decode_strategy}")



class LowBasicModel(pl.LightningModule):

    def __init__(self, args, model, model_name):
        super().__init__()
        self.save_hyperparameters()
        self.optimizer_cfg = args.optimizer
        self.batch_size = None
        self.model = model(vocab=vocab, args=args.model)
        self.configure_optimizers()
        self.model_name = model_name
        self.args = args

    def forward(self, batch):
        gt_obj = batch['object_type']
        gt_action = batch['action_type']
        gt_heatmap = batch['coordinate']

        if self.model_name == "low_policy": 
            pred_obj, pred_action, pred_heatmap = self.model(state=rearrange(batch['state'], 'b h w c -> b c h w'))
        elif self.model_name == "subgoal_low_policy":
            gt_subgoal_action = batch['sub_action']
            gt_subgoal_room = batch['sub_room']
            gt_subgoal_fur = batch['sub_fur']
            gt_subgoal_obj = batch['sub_obj']
            pred_obj, pred_action, pred_heatmap = self.model(state=rearrange(batch['state'], 'b h w c -> b c h w'), action=gt_subgoal_action, obj=gt_subgoal_obj, fur=gt_subgoal_fur, room=gt_subgoal_room)
        else:
            raise NotImplementedError
        return gt_obj, gt_action, gt_heatmap, pred_obj, pred_action, pred_heatmap
        

    def compute_CE_loss(self, gt_label, pred_label, prefix="train/"):
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(pred_label, gt_label)
        self.log(f"{prefix}_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        return loss
    
    def compute_BCE_loss(self, gt_label, pred_label, prefix="train/"):
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(pred_label, gt_label)
        self.log(f"{prefix}_loss", loss.item(), prog_bar=True, batch_size=self.batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        gt_obj, gt_action, gt_heatmap, pred_obj, pred_action, pred_heatmap = self.forward(batch)
        obj_loss = self.compute_CE_loss(gt_obj, pred_obj, prefix="loss/train_obj")
        action_loss = self.compute_CE_loss(gt_action, pred_action, prefix="loss/train_action")
        heatmap_loss = self.compute_BCE_loss(gt_heatmap, pred_heatmap, prefix="loss/train_heatmap")
        loss = obj_loss + action_loss + heatmap_loss
        self.log("loss/train", loss.item(), prog_bar=True, batch_size=self.batch_size)

        # pred_obj = torch.argmax(pred_obj, dim=1)
        # pred_action = torch.argmax(pred_action, dim=1)
        # pred_heatmap = torch.argmax(torch.flatten(pred_heatmap, start_dim=1), dim=1) 
        # correct_objs = pred_obj == gt_obj
        # correct_actions = pred_action == gt_action
        # correct_heatmaps = pred_heatmap == torch.argmax(torch.flatten(gt_heatmap, start_dim=1), dim=1)

        # correct = correct_objs & correct_actions #& correct_heatmaps 
        # num = len(correct) * 1.0
        # metrics['accuracy/train'] = torch.sum(correct).item() / num
        # metrics['accuracy/train_obj'] = torch.sum(correct_objs).item() / num
        # metrics['accuracy/train_action'] = torch.sum(correct_actions).item() / num
        # metrics['accuracy/train_heatmap'] = torch.sum(correct_heatmaps).item() / num

        return loss

    def validation_step(self, batch, batch_idx):
        gt_obj, gt_action, gt_heatmap, pred_obj, pred_action, pred_heatmap = self.forward(batch)
        obj_loss = self.compute_CE_loss(gt_obj, pred_obj, prefix="loss/val_obj")
        action_loss = self.compute_CE_loss(gt_action, pred_action, prefix="loss/val_action")
        heatmap_loss = self.compute_BCE_loss(gt_heatmap, pred_heatmap, prefix="loss/val_heatmap")
        loss = obj_loss + action_loss + heatmap_loss
        self.log("loss/val", loss, prog_bar=True, batch_size=self.batch_size)

        pred_obj = torch.argmax(pred_obj, dim=1)
        pred_action = torch.argmax(pred_action, dim=1)
        pred_heatmap = torch.argmax(torch.flatten(pred_heatmap, start_dim=1), dim=1) 
        correct_objs = pred_obj == gt_obj
        correct_actions = pred_action == gt_action
        correct_heatmaps = pred_heatmap == torch.argmax(torch.flatten(gt_heatmap, start_dim=1), dim=1)

        correct = correct_objs & correct_actions #& correct_heatmaps 
        num = len(correct) * 1.0
        self.log('accuracy/val', torch.sum(correct).item() / num)
        self.log('accuracy/val_obj', torch.sum(correct_objs).item() / num)
        self.log('accuracy/val_action', torch.sum(correct_actions).item() / num)
        # metrics['accuracy/val_heatmap'] = torch.sum(correct_heatmaps).item() / num

    def test_step(self, batch, batch_idx):
        metrics = {}
        gt_obj, gt_action, gt_heatmap, pred_obj, pred_action, pred_heatmap = self.forward(batch)

        obj_loss = self.compute_CE_loss(gt_obj, pred_obj, prefix="loss/test_obj")
        action_loss = self.compute_CE_loss(gt_action, pred_action, prefix="loss/test_action")
        heatmap_loss = self.compute_BCE_loss(gt_heatmap, pred_heatmap, prefix="loss/test_heatmap")
        metrics['loss/test'] = obj_loss + action_loss + heatmap_loss 

        pred_obj = torch.argmax(pred_obj, dim=1)
        pred_action = torch.argmax(pred_action, dim=1)
        pred_heatmap = torch.argmax(torch.flatten(pred_heatmap, start_dim=1), dim=1) 
        correct_objs = pred_obj == gt_obj
        correct_actions = pred_action == gt_action
        correct_heatmaps = pred_heatmap == torch.argmax(torch.flatten(gt_heatmap, start_dim=1), dim=1)

        correct = correct_objs & correct_actions #& correct_heatmaps 
        num = len(correct) * 1.0
        metrics['accuracy/test'] = torch.sum(correct).item() / num
        metrics['accuracy/test_obj'] = torch.sum(correct_objs).item() / num
        metrics['accuracy/test_action'] = torch.sum(correct_actions).item() / num
        metrics['accuracy/test_heatmap'] = torch.sum(correct_heatmaps).item() / num

        self.log_dict(metrics)

        return metrics

    # def configure_optimizers(self):
    #     if 'name' in self.optimizer_cfg:
    #         if self.optimizer_cfg.name == "adam":
    #             optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer_cfg.lr, weight_decay=self.optimizer_cfg.weight_decay) # 1e-5
    #         elif self.optimizer_cfg.name == "adadelta":
    #             optimizer = torch.optim.Adadelta(self.model.parameters())
    #         # ada delta optimizer instead
    #     else:
    #         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer_cfg.lr, weight_decay=self.optimizer_cfg.weight_decay) # 1e-5
    #     return optimizer

    #     # ada delta optimizer instead
    #     return optimizer
    
    def configure_optimizers(self):
        weight_decay = self.optimizer_cfg.weight_decay if "weight_decay" in self.optimizer_cfg else 0
        use_lr_scheduler = self.optimizer_cfg.use_lr_scheduler if "use_lr_scheduler" in self.optimizer_cfg else False
        optimizer = torch.optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr, weight_decay=weight_decay) # 1e-5
        if use_lr_scheduler:
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.optimizer_cfg.lr_restart)
            scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=self.optimizer_cfg.warmup, after_scheduler=scheduler)
            return [optimizer], [{"scheduler": scheduler_warmup, "interval": "epoch"}]
        else:
            return optimizer

    def inference(self, state, state_mask=None, sub_action=None, sub_obj=None, sub_fur=None, sub_room=None):
        B = 1
        if self.model_name == "low_policy":
            state = torch.tensor(state, dtype=torch.float32).to(self.device)  # H, W, C
            if len(state.shape) == 3:
                state = repeat(state, 'h w c -> b c h w', b=B)
            else:
                state = repeat(state, 'b h w c -> b c h w', b=B)
            assert state_mask is None, "state mask must not be provided for normal low policy"

            with torch.no_grad():
                pred_obj_logit, pred_action_logit, pred_heatmap = self.model(state=state)
                pred_obj = torch.argmax(pred_obj_logit, dim=1)  # B, 1
                pred_action = torch.argmax(pred_action_logit, dim=1)  # B, 1
                pred_heatmap = torch.sigmoid(pred_heatmap)
                pred_heatmap = torch.round(pred_heatmap)
        elif self.model_name == "subgoal_low_policy":
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            if len(state.shape) == 3:
                state = repeat(state, 'h w c -> b c h w', b=B)
            else:
                state = repeat(state, 'b h w c -> b c h w', b=B)
            assert sub_action is not None, "action must be provided for subgoal low policy"
            assert sub_obj is not None, "obj must be provided for subgoal low policy"
            assert sub_fur is not None, "fur must be provided for subgoal low policy"
            assert sub_room is not None, "room must be provided for subgoal low policy"
            sub_action = torch.tensor(sub_action, dtype=torch.long).to(self.device)
            sub_obj = torch.tensor(sub_obj, dtype=torch.long).to(self.device)
            sub_fur = torch.tensor(sub_fur, dtype=torch.long).to(self.device)
            sub_room = torch.tensor(sub_room, dtype=torch.long).to(self.device)
            with torch.no_grad():
                pred_obj_logit, pred_action_logit, pred_heatmap = self.model(state=state, action=sub_action, obj=sub_obj, fur=sub_fur, room=sub_room)
                pred_obj = torch.argmax(pred_obj_logit, dim=1)
                pred_action = torch.argmax(pred_action_logit, dim=1)
                pred_heatmap = torch.sigmoid(pred_heatmap)
                pred_heatmap = torch.round(pred_heatmap)
        else:
            raise NotImplementedError
            # TODO: implement other models

        return pred_obj, pred_action, pred_heatmap

    def inference_sample(self, state, vocab, state_mask=None, sub_action=None, sub_obj=None, sub_fur=None, sub_room=None, temp=1, use_audio=None):
        # TODO: clean up 
        if self.model_name == "low_policy": 
            state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)  # H, W, C
            state = torch.squeeze(state)
            # print('shape of state, num_samples x state shape', state.shape)
            if len(state.shape) == 3:
                state = repeat(state, 'h w c -> b c h w', b=1)
            else:
                state = repeat(state, 'b h w c -> b c h w')#, b=1)
            assert state_mask is None, "state mask must not be provided for normal low policy"

            with torch.no_grad(): 
                pred_obj_logit, pred_action_logit, pred_heatmap = self.model(state=state)
                pred_obj_logit = torch.softmax(pred_obj_logit / temp, 1)

                if use_audio is not None:
                    print('use_audio')
                    pred_action_logit = torch.where(torch.tensor(use_audio).to(self.device) > 0, pred_action_logit, -np.inf)
                    # TODO: get a distribution based on dataset
                # temp = 1.5
                pred_action_logit = torch.softmax(pred_action_logit / temp, 1)
                pred_obj = torch.multinomial(pred_obj_logit, 1).cpu()
                pred_action = torch.multinomial(pred_action_logit, 1).cpu()
                pred_heatmap = torch.round(torch.sigmoid(pred_heatmap)).cpu()
        elif self.model_name == 'subgoal_low_policy':
            state = torch.tensor(np.array(state), dtype=torch.float32).to(self.device)
            state = torch.squeeze(state)

            if len(state.shape) == 3:
                state = repeat(state, 'h w c -> b c h w', b=1)
            else:
                state = repeat(state, 'b h w c -> b c h w')

            assert sub_action is not None, "action must be provided for subgoal low policy"
            assert sub_obj is not None, "obj must be provided for subgoal low policy"
            assert sub_fur is not None, "fur must be provided for subgoal low policy"
            assert sub_room is not None, "room must be provided for subgoal low policy"

            sub_action = torch.tensor(sub_action, dtype=torch.long).to(self.device) #.unsqueeze(0) # TODO: debug one_sample
            sub_obj = torch.tensor(sub_obj, dtype=torch.long).to(self.device) #.unsqueeze(0) # TODO: debug one_sample
            sub_fur = torch.tensor(sub_fur, dtype=torch.long).to(self.device) #.unsqueeze(0) # TODO: debug one_sample
            sub_room = torch.tensor(sub_room, dtype=torch.long).to(self.device) #.unsqueeze(0) # TODO: debug one_sample

            with torch.no_grad():
                pred_obj_logit, pred_action_logit, pred_heatmap = self.model(state=state, obj=sub_obj, fur=sub_fur, room=sub_room, action=sub_action)
                pred_obj_logit = torch.softmax(pred_obj_logit / temp, 1)
                pred_action_logit = torch.softmax(pred_action_logit / temp, 1)

                pred_obj = torch.multinomial(pred_obj_logit, 1).cpu()
                pred_action = torch.multinomial(pred_action_logit, 1).cpu()
                pred_heatmap = torch.round(torch.sigmoid(pred_heatmap)).cpu()    

            # breakpoint()            
        else:
            raise NotImplementedError
            # TODO: implement other models

        return pred_obj, pred_action, pred_heatmap
    
    def inference_list(self, state, vocab, num_samples = 20, sub_action=None, sub_obj=None, sub_fur=None, sub_room=None):
        '''
        Returns list of len num_samples of zipped objects, actions, and heatmaps with beam search.
        '''
        B = 1
        state = torch.tensor(state, dtype=torch.float32).to(self.device)  # H, W, C
        state = repeat(state, 'h w c -> b c h w', b=B)

        if sub_action:
            sub_action = torch.full(
                (B,), sub_action, dtype=torch.long).to(self.device)
        if sub_obj:
            sub_obj = torch.full(
                (B,), sub_obj, dtype=torch.long).to(self.device)
        if sub_fur:
            sub_fur = torch.full(
                (B,), sub_fur, dtype=torch.long).to(self.device)
        if sub_room:
            sub_room = torch.full(
                (B,), sub_room, dtype=torch.long).to(self.device)

        if self.model_name == "low_policy":
            with torch.no_grad():
                pred_obj_logit, pred_action_logit, pred_heatmap = self.model(state=state)
        elif self.model_name == "subgoal_low_policy":
            assert sub_action is not None, "action must be provided for subgoal low policy"
            assert sub_obj is not None, "obj must be provided for subgoal low policy"
            assert sub_fur is not None, "fur must be provided for subgoal low policy"
            assert sub_room is not None, "room must be provided for subgoal low policy"
            with torch.no_grad():
                pred_obj_logit, pred_action_logit, pred_heatmap = self.model(state=state, action=sub_action, obj=sub_obj, fur=sub_fur, room=sub_room)
        else:
            raise NotImplementedError
        
        pred_action_prob = F.softmax(pred_action_logit, dim=1)
        k_a = min(num_samples, len(vocab.ACTION_TO_IDX))
        sampled_action_prob, sampled_action = torch.topk(pred_action_prob[0], k=k_a)  # k_a

        pred_obj_prob = F.softmax(pred_obj_logit, dim=1)
        pred_ao_prob = repeat(sampled_action_prob, 'k -> k v', v=len(vocab.OBJECT_TO_IDX)) * pred_obj_prob
        k_ao = min(num_samples, len(vocab.ACTION_TO_IDX) * len(vocab.OBJECT_TO_IDX))
        sampled_ao_prob, sampled_ao = torch.topk(pred_ao_prob.view(-1), k=k_ao)
        sampled_a_idx = sampled_ao // len(vocab.OBJECT_TO_IDX)  # k
        sampled_obj = sampled_ao % len(vocab.OBJECT_TO_IDX)  # k
        sampled_action = sampled_action[sampled_a_idx]  # k

        pred_heatmap = torch.sigmoid(pred_heatmap)
        pred_heatmap = torch.round(pred_heatmap)
        
        sampled_heatmap = repeat(pred_heatmap, 'B h w -> x B h w', x = k_ao)
        sampled_heatmap = sampled_heatmap.squeeze(1)

        return sampled_obj, sampled_action, sampled_heatmap, sampled_ao_prob
