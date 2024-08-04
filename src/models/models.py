import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import Transformer
from einops import rearrange, repeat, reduce

from src.models.vit import SimpleViT, PerPatchViT

def generate_square_subsequent_mask(sz):
    """
    Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    e.g.,
    tensor([[0., -inf, -inf],
            [0., 0., -inf],
            [0., 0., 0.]])
    """
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PolicyModel(torch.nn.Module):

    def __init__(self, vocab, args, num_classes=1000):
        super(PolicyModel, self).__init__()
        self.encoder = SimpleViT(
                            image_size=(20, 20),
                            patch_size=args.vit_patch_size,
                            num_classes=num_classes, # what's this 1000 n classes?
                            dim=args.vit_dim,
                            depth=args.vit_depth,
                            heads=args.vit_heads,
                            mlp_dim=args.vit_mlp_dim,
                            channels=args.vit_channels,
                        )
        
        self.object_decoder = nn.Sequential(nn.Linear(num_classes, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, len(vocab.OBJECT_TO_IDX)))
        
        self.furniture_decoder = nn.Sequential(nn.Linear(num_classes, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, len(vocab.OBJECT_TO_IDX)))
        
        # object and furniture share the same vocab because object can also be furniture in subgoals
        self.room_decoder = nn.Sequential(nn.Linear(num_classes, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, len(vocab.ROOM_TO_IDX)))
        self.action_decoder = nn.Sequential(nn.Linear(num_classes, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, len(vocab.ACTION_TO_IDX)))

    def forward(self, state):

        embedding = self.encoder(state)
        obj = self.object_decoder(embedding)
        fur = self.furniture_decoder(embedding)
        room = self.room_decoder(embedding)
        action = self.action_decoder(embedding)

        return obj, fur, room, action


class TransformerHeadPolicyModel(torch.nn.Module):

    def __init__(self, vocab, args):
        super(TransformerHeadPolicyModel, self).__init__()
        embedding_dim = args.transformer_input_dim - args.transformer_position_embedding_dim

        self.vit_encoder = SimpleViT(
            image_size=(20, 20),
            patch_size=args.vit_patch_size,
            num_classes=embedding_dim,
            dim=args.vit_dim,
            depth=args.vit_depth,
            heads=args.vit_heads,
            mlp_dim=args.vit_mlp_dim,
            channels=args.vit_channels
        )
        # autoregressively predict action, room, furniture, object
        assert embedding_dim > 0

        # embeddings
        self.object_embedding = torch.nn.Embedding(len(vocab.OBJECT_TO_IDX), embedding_dim)
        # object and furniture share the same vocab because object can also be furniture in subgoals
        self.furniture_embedding = torch.nn.Embedding(len(vocab.OBJECT_TO_IDX), embedding_dim)
        self.room_embedding = torch.nn.Embedding(len(vocab.ROOM_TO_IDX), embedding_dim)
        self.action_embedding = torch.nn.Embedding(len(vocab.ACTION_TO_IDX), embedding_dim)

        # auxillary embeddings
        self.start_token_embeddings = torch.nn.Embedding(1, embedding_dim)
        self.position_embeddings = torch.nn.Embedding(args.transformer_position_embedding_dim, args.transformer_position_embedding_dim) # should this be eg 4 by 4?

        # transformer prediction head
        self.output_head = Transformer(d_model=args.transformer_input_dim,
                                       dim_feedforward=args.transformer_hidden_dim,
                                       nhead=args.transformer_n_heads,
                                       num_encoder_layers=args.transformer_depth,
                                       num_decoder_layers=args.transformer_depth, 
                                       dropout=args.transformer_dropout,
                                       activation=args.transformer_activation,
                                       batch_first=True)

        self.output_downscale = torch.nn.Linear(args.transformer_input_dim, embedding_dim)

    def forward(self, state, obj, fur, room, action, **kwargs):
        B, C, H, W = state.shape
        device = state.device
        # encode state
        vision_embed = rearrange(self.vit_encoder(state), 'B H -> B 1 H')
        # embed
        obj_embed = rearrange(self.object_embedding(obj), 'B H -> B 1 H')
        fur_embed = rearrange(self.furniture_embedding(fur), 'B H -> B 1 H')
        room_embed = rearrange(self.room_embedding(room), 'B H -> B 1 H')
        action_embed = rearrange(self.action_embedding(action), 'B H -> B 1 H')

        # position embeddings
        pos_idx = repeat(torch.arange(4, device=device), 'L -> B L', B=B)
        pos_embed = self.position_embeddings(pos_idx)  # B, 4, pos_dim

        # start token
        start_token = repeat(self.start_token_embeddings(torch.arange(1, device=device)), 'L H -> B L H', B=B)  # B, 1, some dim

        # build input and output sequences
        src_sequence_encode = torch.cat([vision_embed, pos_embed[:, 0].unsqueeze(1)], dim=-1)  # B, 1, some dim

        # tgt_sequence_encode = torch.cat([start_token, action_embed, room_embed, fur_embed], dim=1)  # B, 4, some dim
        tgt_sequence_encode = torch.cat([start_token, room_embed, fur_embed, obj_embed], dim=1)  # B, 4, some dim
        tgt_sequence_encode = torch.cat([tgt_sequence_encode, pos_embed], dim=-1)  # B, 4, some dim

        # build causal attention mask
        tgt_mask = generate_square_subsequent_mask(4).to(device)  # 4, 4

        # transformer prediction
        # predict action based on learned start token
        # predict room based on learned start token, action
        # predict furniture based on learned start token, action, room
        # predict object based on learned start token, action, room, furniture

        # print(src_sequence_encode.shape)
        # print(tgt_sequence_encode.shape)

        output = self.output_head(src=src_sequence_encode, tgt=tgt_sequence_encode, tgt_mask=tgt_mask)  # B, 4, some dim

        # we want to make sure the transformer output is the same size as the embedding
        output = self.output_downscale(output)

        # separate outputs
        # action_encode = output[:, 0]  # B, some dim
        # room_encode = output[:, 1]  # B, some dim
        # fur_encode = output[:, 2]  # B, some dim
        # obj_encode = output[:, 3]  # B, some dim
        room_encode = output[:, 0]  # B, some dim
        fur_encode = output[:, 1]  # B, some dim
        obj_encode = output[:, 2]  # B, some dim
        action_encode = output[:, 3]  # B, some dim        

        # obj_encode: B, H
        # object_embedding.weight: V, H
        # obj: B, V
        obj = torch.mm(obj_encode, self.object_embedding.weight.transpose(1, 0))
        fur = torch.mm(fur_encode, self.furniture_embedding.weight.transpose(1, 0))
        room = torch.mm(room_encode, self.room_embedding.weight.transpose(1, 0))
        action = torch.mm(action_encode, self.action_embedding.weight.transpose(1, 0))

        return obj, fur, room, action


class HistAwareTransformerHeadPolicyModel(torch.nn.Module):

    def __init__(self, vocab, args):
        super(HistAwareTransformerHeadPolicyModel, self).__init__()
        embedding_dim = args.transformer_input_dim - args.transformer_position_embedding_dim

        self.vit_encoder = SimpleViT(
            image_size=(20, 20),
            patch_size=args.vit_patch_size,
            num_classes=embedding_dim,
            dim=args.vit_dim,
            depth=args.vit_depth,
            heads=args.vit_heads,
            mlp_dim=args.vit_mlp_dim,
            channels=args.vit_channels
        )
        # autoregressively predict action, room, furniture, object
        assert embedding_dim > 0

        # embeddings
        self.object_embedding = torch.nn.Embedding(len(vocab.OBJECT_TO_IDX), embedding_dim)
        # object and furniture share the same vocab because object can also be furniture in subgoals
        self.furniture_embedding = torch.nn.Embedding(len(vocab.OBJECT_TO_IDX), embedding_dim)
        self.room_embedding = torch.nn.Embedding(len(vocab.ROOM_TO_IDX), embedding_dim)
        self.action_embedding = torch.nn.Embedding(len(vocab.ACTION_TO_IDX), embedding_dim)

        # auxillary embeddings
        self.start_token_embeddings = torch.nn.Embedding(1, embedding_dim)
        self.position_embeddings = torch.nn.Embedding(100, args.transformer_position_embedding_dim)

        # transformer prediction head
        self.output_head = Transformer(d_model=args.transformer_input_dim,
                                       dim_feedforward=args.transformer_hidden_dim,
                                       nhead=args.transformer_n_heads,
                                       num_encoder_layers=args.transformer_depth,
                                       num_decoder_layers=args.transformer_depth,
                                       dropout=args.transformer_dropout,
                                       activation=args.transformer_activation,
                                       batch_first=True)

        self.output_downscale = torch.nn.Linear(args.transformer_input_dim, embedding_dim)

    def forward(self, state, obj, fur, room, action, state_mask):

        # T is history length
        B, T, C, H, W = state.shape
        device = state.device

        # encode state
        vision_embed = rearrange(self.vit_encoder(rearrange(state, 'B T C H W -> (B T) C H W')), '(B T) H -> B T H', B=B)
        # embed
        obj_embed = rearrange(self.object_embedding(obj), 'B H -> B 1 H')
        fur_embed = rearrange(self.furniture_embedding(fur), 'B H -> B 1 H')
        room_embed = rearrange(self.room_embedding(room), 'B H -> B 1 H')
        action_embed = rearrange(self.action_embedding(action), 'B H -> B 1 H')

        # position embeddings
        tgt_pos_idx = repeat(torch.arange(4, device=device), 'L -> B L', B=B)
        src_pos_idx = repeat(torch.arange(T, device=device), 'T -> B T', B=B)
        tgt_pos_embed = self.position_embeddings(tgt_pos_idx)  # B, L, pos_dim
        src_pos_embed = self.position_embeddings(src_pos_idx)  # B, T, pos_dim

        # start token
        start_token = repeat(self.start_token_embeddings(torch.arange(1, device=device)), 'L H -> B L H', B=B)  # B, 1, some dim

        # build input and output sequences
        src_sequence_encode = torch.cat([vision_embed, src_pos_embed], dim=-1)  # B, T, some dim

        tgt_sequence_encode = torch.cat([start_token, action_embed, room_embed, fur_embed], dim=1)  # B, 4, some dim
        # tgt_sequence_encode = torch.cat([start_token, room_embed, fur_embed, obj_embed], dim=1)  # B, 4, some dim
        tgt_sequence_encode = torch.cat([tgt_sequence_encode, tgt_pos_embed], dim=-1)  # B, 4, some dim

        # build causal attention mask
        tgt_mask = generate_square_subsequent_mask(4).to(device)  # 4, 4

        # transformer prediction
        # predict action based on learned start token
        # predict room based on learned start token, action
        # predict furniture based on learned start token, action, room
        # predict object based on learned start token, action, room, furniture

        # print(src_sequence_encode.shape)
        # print(tgt_sequence_encode.shape)

        # state_mask: B, T
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        # therefore, for padding positions, we want to set the mask to 1
        state_mask = (state_mask == 1)
        output = self.output_head(src=src_sequence_encode, tgt=tgt_sequence_encode, tgt_mask=tgt_mask, src_key_padding_mask=state_mask)  # B, 4, some dim

        # we want to make sure the transformer output is the same size as the embedding
        output = self.output_downscale(output)

        # separate outputs
        action_encode = output[:, 0]  # B, some dim
        room_encode = output[:, 1]  # B, some dim
        fur_encode = output[:, 2]  # B, some dim
        obj_encode = output[:, 3]  # B, some dim
        #room_encode = output[:, 0]  # B, some dim
        #fur_encode = output[:, 1]  # B, some dim
        #obj_encode = output[:, 2]  # B, some dim        
        #action_encode = output[:, 3]  # B, some dim

        # obj_encode: B, H
        # object_embedding.weight: V, H
        # obj: B, V
        obj = torch.mm(obj_encode, self.object_embedding.weight.transpose(1, 0))
        fur = torch.mm(fur_encode, self.furniture_embedding.weight.transpose(1, 0))
        room = torch.mm(room_encode, self.room_embedding.weight.transpose(1, 0))
        action = torch.mm(action_encode, self.action_embedding.weight.transpose(1, 0))

        return obj, fur, room, action


class AudioConditionedTransformerHeadPolicyModel(torch.nn.Module):

    # additional args for audio model:
    # max_audio_length

    def __init__(self, vocab, args):
        super(AudioConditionedTransformerHeadPolicyModel, self).__init__()
        embedding_dim = args.transformer_input_dim - args.transformer_position_embedding_dim

        self.vit_encoder = SimpleViT(
            image_size=(25, 25),
            patch_size=args.vit_patch_size,
            num_classes=embedding_dim,
            dim=args.vit_dim,
            depth=args.vit_depth,
            heads=args.vit_heads,
            mlp_dim=args.vit_mlp_dim,
            channels=args.vit_channels
        )

        audio_hidden_dim = embedding_dim * 2
        self.audio_encoder = nn.Sequential(
            nn.Linear(args.max_audio_length, audio_hidden_dim),
            nn.LayerNorm(audio_hidden_dim),
            nn.ReLU(),
            nn.Linear(audio_hidden_dim, audio_hidden_dim),
            nn.LayerNorm(audio_hidden_dim),
            nn.ReLU(),
            nn.Linear(audio_hidden_dim, embedding_dim))

        # autoregressively predict action, room, furniture, object
        assert embedding_dim > 0

        # embeddings
        self.object_embedding = torch.nn.Embedding(len(vocab.OBJECT_TO_IDX), embedding_dim)
        # object and furniture share the same vocab because object can also be furniture in subgoals
        self.furniture_embedding = torch.nn.Embedding(len(vocab.OBJECT_TO_IDX), embedding_dim)
        self.room_embedding = torch.nn.Embedding(len(vocab.ROOM_TO_IDX), embedding_dim)
        self.action_embedding = torch.nn.Embedding(len(vocab.ACTION_TO_IDX), embedding_dim)

        # auxillary embeddings
        self.start_token_embeddings = torch.nn.Embedding(1, embedding_dim)
        self.position_embeddings = torch.nn.Embedding(args.transformer_position_embedding_dim, args.transformer_position_embedding_dim) # should this be eg 4 by 4?

        # transformer prediction head
        self.output_head = Transformer(d_model=args.transformer_input_dim,
                                       dim_feedforward=args.transformer_hidden_dim,
                                       nhead=args.transformer_n_heads,
                                       num_encoder_layers=args.transformer_depth,
                                       num_decoder_layers=args.transformer_depth,
                                       dropout=args.transformer_dropout,
                                       activation=args.transformer_activation,
                                       batch_first=True)

        self.output_downscale = torch.nn.Linear(args.transformer_input_dim, embedding_dim)

    def forward(self, state, audio, audio_mask, obj, fur, room, action, **kwargs):
        # audio: B, A
        # audio_mask: B, A

        B, C, H, W = state.shape

        B, A = audio.shape  # A is max audio length

        device = state.device
        # encode state
        vision_embed = rearrange(self.vit_encoder(state), 'B H -> B 1 H')

        audio_embed = rearrange(self.audio_encoder(audio), 'B H -> B 1 H')  # B, H

        # embed
        # obj_embed = rearrange(self.object_embedding(obj), 'B H -> B 1 H')
        fur_embed = rearrange(self.furniture_embedding(fur), 'B H -> B 1 H')
        room_embed = rearrange(self.room_embedding(room), 'B H -> B 1 H')
        action_embed = rearrange(self.action_embedding(action), 'B H -> B 1 H')

        # position embeddings
        pos_idx = repeat(torch.arange(4, device=device), 'L -> B L', B=B)
        pos_embed = self.position_embeddings(pos_idx)  # B, 4, pos_dim

        # start token
        start_token = repeat(self.start_token_embeddings(torch.arange(1, device=device)), 'L H -> B L H', B=B)  # B, 1, some dim

        # build input and output sequences
        src_sequence_encode = torch.cat([vision_embed, audio_embed], dim=1)  # B, 2, some dim
        src_sequence_encode = torch.cat([src_sequence_encode, pos_embed[:, :2]], dim=-1)  # B, 2, some dim

        tgt_sequence_encode = torch.cat([start_token, action_embed, room_embed, fur_embed], dim=1)  # B, 4, some dim
        tgt_sequence_encode = torch.cat([tgt_sequence_encode, pos_embed], dim=-1)  # B, 4, some dim

        # build causal attention mask
        tgt_mask = generate_square_subsequent_mask(4).to(device)  # 4, 4

        # transformer prediction
        # predict action based on learned start token
        # predict room based on learned start token, action
        # predict furniture based on learned start token, action, room
        # predict object based on learned start token, action, room, furniture

        # print(src_sequence_encode.shape)
        # print(tgt_sequence_encode.shape)

        output = self.output_head(src=src_sequence_encode, tgt=tgt_sequence_encode, tgt_mask=tgt_mask)  # B, 4, some dim

        # we want to make sure the transformer output is the same size as the embedding
        output = self.output_downscale(output)

        # separate outputs
        action_encode = output[:, 0]  # B, some dim
        room_encode = output[:, 1]  # B, some dim
        fur_encode = output[:, 2]  # B, some dim
        obj_encode = output[:, 3]  # B, some dim

        # obj_encode: B, H
        # object_embedding.weight: V, H
        # obj: B, V
        obj = torch.mm(obj_encode, self.object_embedding.weight.transpose(1, 0))
        fur = torch.mm(fur_encode, self.furniture_embedding.weight.transpose(1, 0))
        room = torch.mm(room_encode, self.room_embedding.weight.transpose(1, 0))
        action = torch.mm(action_encode, self.action_embedding.weight.transpose(1, 0))

        return obj, fur, room, action


class LowPolicyModel(torch.nn.Module):

    def __init__(self, vocab, args):
        super(LowPolicyModel, self).__init__()
        embedding_dim = args.embedding_dim # need to set in args, 128/256

        self.vit_encoder = PerPatchViT(
            image_size=tuple(args.vit_image_size),
            patch_size=args.vit_patch_size,
            num_classes=embedding_dim,
            dim=args.vit_dim,
            depth=args.vit_depth,
            heads=args.vit_heads,
            mlp_dim=args.vit_mlp_dim,
            channels=args.vit_channels
        )
        
        # autoregressively predict action_type, target_type
        assert embedding_dim > 0

        # MLP for heatmap predction
        #TODO: check if input dim args.vit_dim is correct
        self.heatmap_mlp = nn.Sequential(
            nn.Linear(args.vit_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),            
            nn.Linear(embedding_dim, 1)
        )

        # MLP for action type prediction
        self.action_type_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),         
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),                     
            nn.Linear(embedding_dim, len(vocab.ACTION_TO_IDX))
        )

        # MLP for target type prediction
        self.target_type_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),            
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),               
            nn.Linear(embedding_dim, len(vocab.OBJECT_TO_IDX))
        )

    def forward(self, state, **kwargs):
        B, C, H, W = state.shape
        device = state.device

        # encode state
        mean_vision_embed, per_patch_vision_embed = self.vit_encoder(state) # return B embedding_dim, B (H W) args.vit_dim
        # target action prediction head
        target = self.target_type_mlp(mean_vision_embed)
        action = self.action_type_mlp(mean_vision_embed)
        
        # heatmap prediction head
        #TODO: check if rearrange after heatmap_mlp is correct
        heatmap = self.heatmap_mlp(per_patch_vision_embed)
        heatmap = rearrange(heatmap, 'B (H W) 1 -> B H W', H=H, W=W)  # B (H W) 1 -> B H W
         # B (H, W) E -> B (H W)  # binary cross entropy loss
        return target, action, heatmap


class SubgoalConditionedLowPolicyModel(torch.nn.Module):

    def __init__(self, vocab, args):
        super(SubgoalConditionedLowPolicyModel, self).__init__()
        embedding_dim = args.embedding_dim # need to set in args, 128/256
        subgoal_channel = args.subgoal_channel # need to set in args, 4

        self.vit_encoder = PerPatchViT(
            image_size=(20, 20),
            patch_size=args.vit_patch_size,
            num_classes=embedding_dim,
            dim=args.vit_dim,
            depth=args.vit_depth,
            heads=args.vit_heads,
            mlp_dim=args.vit_mlp_dim,
            channels=args.vit_channels + args.subgoal_channel
        )
        
        # autoregressively predict action_type, target_type
        assert embedding_dim > 0

        # MLP for heatmap predction
        #TODO: check if input dim args.vit_dim is correct
        self.heatmap_mlp = nn.Sequential(
            nn.Linear(args.vit_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, 1)
        )

        # MLP for action type prediction
        self.action_type_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, len(vocab.ACTION_TO_IDX))
        )

        # MLP for target type prediction
        self.target_type_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, len(vocab.OBJECT_TO_IDX))
        )

        # embeddings for subgoal conditioning
        self.object_embedding = torch.nn.Embedding(len(vocab.OBJECT_TO_IDX), embedding_dim)
        # object and furniture share the same vocab because object can also be furniture in subgoals
        self.furniture_embedding = torch.nn.Embedding(len(vocab.OBJECT_TO_IDX), embedding_dim)
        self.room_embedding = torch.nn.Embedding(len(vocab.ROOM_TO_IDX), embedding_dim)
        self.action_embedding = torch.nn.Embedding(len(vocab.ACTION_TO_IDX), embedding_dim)

        self.subgoal_conditioning_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, subgoal_channel)
        )

    def forward(self, state, action, obj, fur, room, **kwargs):
        B, C, H, W = state.shape
        device = state.device
        # encode subgoal
        object_embed = self.object_embedding(obj)#.squeeze() # 'B H' 
        furniture_embed = self.furniture_embedding(fur)#.squeeze() # 'B H'
        room_embed = self.room_embedding(room)#.squeeze()# 'B H'
        action_embed = self.action_embedding(action)#.squeeze() # 'B H'

        subgoal_embed = torch.cat((object_embed, furniture_embed, room_embed, action_embed), 1) # 'B 4H'
        subgoal_embed = self.subgoal_conditioning_mlp(subgoal_embed) # 'B SUBGOAL_C'
        
        # encode state + subgoal
        subgoal_embed = repeat(subgoal_embed, 'B C -> B C H W', H=H, W=W) # 'B SUBGOAL_C H W'
        state_subgoal = torch.cat((state, subgoal_embed), dim=1) # 'B C+SUBGOAL_C H W'
        mean_vision_embed, per_patch_vision_embed = self.vit_encoder(state_subgoal) # B embedding_dim, B (H W) args.vit_dim

        # target action prediction head
        target = self.target_type_mlp(mean_vision_embed)
        action = self.action_type_mlp(mean_vision_embed)
        
        # heatmap prediction head
        #TODO: check if rearrange after heatmap_mlp is correct
        heatmap = self.heatmap_mlp(per_patch_vision_embed)
        heatmap = rearrange(heatmap, 'B (H W) 1 -> B H W', H=H, W=W)  # B (H W) 1 -> B H W
         # B (H, W) E -> B (H W)  # binary cross entropy loss

        return target, action, heatmap


if __name__ == "__main__":

    # # test model
    # import argparse
    # import MarpleLongModels.models.vocab as vocab
    #
    # args = argparse.Namespace()
    # args.vit_patch_dim = 32
    # args.vit_depth = 3
    # args.vit_n_heads = 4
    #
    # args.position_embedding_dim = 4
    #
    # args.transformer_input_dim = 32
    # args.transformer_n_heads = 4
    # args.transformer_depth = 3
    # args.transformer_hidden_dim = 32
    # args.transformer_dropout = 0.1
    # args.transformer_activation = 'gelu'
    #
    # model = TransformerHeadPolicyModel(vocab, args)
    # # print(model)
    #
    # B = 32
    # state = torch.randn(B, 8, 25, 25)
    # obj = torch.randint(0, 2, (B, ))
    # fur = torch.randint(0, 2, (B, ))
    # room = torch.randint(0, 2, (B, ))
    # action = torch.randint(0, 2, (B, ))
    #
    # obj, fur, room, action = model(state, obj, fur, room, action)
    # print(obj.shape)
    # print(fur.shape)
    # print(room.shape)
    # print(action.shape)

    # test history aware model
    import argparse
    import MarpleLongModels.models.vocab as vocab

    args = argparse.Namespace()
    args.vit_dim = 32  # patch dim
    args.vit_depth = 3
    args.vit_heads = 4
    args.vit_patch_size = 1
    args.vit_mlp_dim = 28  # args.transformer_input_dim - args.transformer_position_embedding_dim
    args.vit_channels = 8

    args.transformer_input_dim = 32
    args.transformer_n_heads = 4
    args.transformer_depth = 3
    args.transformer_hidden_dim = 32
    args.transformer_dropout = 0.1
    args.transformer_activation = 'gelu'
    args.transformer_position_embedding_dim = 4

    model = HistAwareTransformerHeadPolicyModel(vocab, args)
    # print(model)

    B = 32
    T = 6
    state = torch.randn(B, 6, 8, 25, 25)
    obj = torch.randint(0, 2, (B, ))
    fur = torch.randint(0, 2, (B, ))
    room = torch.randint(0, 2, (B, ))
    action = torch.randint(0, 2, (B, ))
    state_mask = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.bool).repeat(B, 1)

    obj, fur, room, action = model(state, obj, fur, room, action, state_mask)
    # print(obj.shape)
    # print(fur.shape)
    # print(room.shape)
    # print(action.shape)
