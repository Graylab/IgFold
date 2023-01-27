import os
from einops import rearrange, repeat
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch3d.transforms import quaternion_to_matrix

from igfold.model.interface import *
from igfold.model.components import TriangleGraphTransformer, IPAEncoder, IPATransformer
from igfold.utils.coordinates import get_ideal_coords, place_o_coords
from igfold.training.utils import *
from igfold.utils.general import exists

ATOM_DIM = 3


class IgFold(pl.LightningModule):
    def __init__(
        self,
        config,
        config_overwrite=None,
    ):
        super().__init__()

        import transformers

        self.save_hyperparameters()
        config = self.hparams.config
        if exists(config_overwrite):
            config.update(config_overwrite)

        self.tokenizer = config["tokenizer"]
        self.vocab_size = len(self.tokenizer.vocab)
        self.bert_model = transformers.BertModel(config["bert_config"])
        bert_layers = self.bert_model.config.num_hidden_layers
        self.bert_feat_dim = self.bert_model.config.hidden_size
        self.bert_attn_dim = bert_layers * self.bert_model.config.num_attention_heads

        self.node_dim = config["node_dim"]

        self.depth = config["depth"]
        self.gt_depth = config["gt_depth"]
        self.gt_heads = config["gt_heads"]

        self.temp_ipa_depth = config["temp_ipa_depth"]
        self.temp_ipa_heads = config["temp_ipa_heads"]

        self.str_ipa_depth = config["str_ipa_depth"]
        self.str_ipa_heads = config["str_ipa_heads"]

        self.dev_ipa_depth = config["dev_ipa_depth"]
        self.dev_ipa_heads = config["dev_ipa_heads"]

        self.str_node_transform = nn.Sequential(
            nn.Linear(
                self.bert_feat_dim,
                self.node_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim),
        )
        self.str_edge_transform = nn.Sequential(
            nn.Linear(
                self.bert_attn_dim,
                self.node_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim),
        )

        self.main_block = TriangleGraphTransformer(
            dim=self.node_dim,
            edge_dim=self.node_dim,
            depth=self.depth,
            tri_dim_hidden=2 * self.node_dim,
            gt_depth=self.gt_depth,
            gt_heads=self.gt_heads,
            gt_dim_head=self.node_dim // 2,
        )
        self.template_ipa = IPAEncoder(
            dim=self.node_dim,
            depth=self.temp_ipa_depth,
            heads=self.temp_ipa_heads,
            require_pairwise_repr=True,
        )

        self.structure_ipa = IPATransformer(
            dim=self.node_dim,
            depth=self.str_ipa_depth,
            heads=self.str_ipa_heads,
            require_pairwise_repr=True,
        )

        self.dev_node_transform = nn.Sequential(
            nn.Linear(self.bert_feat_dim, self.node_dim),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim),
        )
        self.dev_edge_transform = nn.Sequential(
            nn.Linear(
                self.bert_attn_dim,
                self.node_dim,
            ),
            nn.ReLU(),
            nn.LayerNorm(self.node_dim),
        )
        self.dev_ipa = IPAEncoder(
            dim=self.node_dim,
            depth=self.dev_ipa_depth,
            heads=self.dev_ipa_heads,
            require_pairwise_repr=True,
        )
        self.dev_linear = nn.Linear(
            self.node_dim,
            4,
        )

    def get_tokens(
        self,
        seq,
    ):
        if isinstance(seq, str):
            tokens = self.tokenizer.encode(
                " ".join(list(seq)),
                return_tensors="pt",
            )
        elif isinstance(seq, list) and isinstance(seq[0], str):
            seqs = [" ".join(list(s)) for s in seq]
            tokens = self.tokenizer.batch_encode_plus(
                seqs,
                return_tensors="pt",
            )["input_ids"]
        else:
            tokens = seq

        return tokens.to(self.device)

    def get_bert_feats(self, tokens):
        bert_output = self.bert_model(
            tokens,
            output_hidden_states=True,
            output_attentions=True,
        )

        feats = bert_output.hidden_states[-1]
        feats = feats[:, 1:-1]

        attn = torch.cat(
            bert_output.attentions,
            dim=1,
        )
        attn = attn[:, :, 1:-1, 1:-1]
        attn = rearrange(
            attn,
            "b d i j -> b i j d",
        )

        hidden = bert_output.hidden_states

        return feats, attn, hidden

    def get_coords_tran_rot(
        self,
        temp_coords,
        batch_size,
        seq_len,
    ):
        res_coords = rearrange(
            temp_coords,
            "b (l a) d -> b l a d",
            l=seq_len,
        ).to(self.device)
        ideal_coords = get_ideal_coords()
        res_ideal_coords = repeat(
            ideal_coords,
            "a d -> b l a d",
            b=batch_size,
            l=seq_len,
        ).to(self.device)
        _, rotations, translations = kabsch(
            res_ideal_coords,
            res_coords,
            return_translation_rotation=True,
        )
        translations = rearrange(
            translations,
            "b l () d -> b l d",
        )

        return translations, rotations

    def clean_input(
        self,
        input: IgFoldInput,
    ):
        tokens = [self.get_tokens(s) for s in input.sequences]

        temp_coords = input.template_coords
        temp_mask = input.template_mask
        batch_mask = input.batch_mask
        align_mask = input.align_mask

        batch_size = tokens[0].shape[0]
        seq_lens = [max(t.shape[1] - 2, 0) for t in tokens]
        seq_len = sum(seq_lens)

        if not exists(temp_coords):
            temp_coords = torch.zeros(
                batch_size,
                4 * seq_len,
                ATOM_DIM,
                device=self.device,
            ).float()
        if not exists(temp_mask):
            temp_mask = torch.zeros(
                batch_size,
                4 * seq_len,
                device=self.device,
            ).bool()
        if not exists(batch_mask):
            batch_mask = torch.ones(
                batch_size,
                4 * seq_len,
                device=self.device,
            ).bool()
        if not exists(align_mask):
            align_mask = torch.ones(
                batch_size,
                4 * seq_len,
                device=self.device,
            ).bool()

        align_mask = align_mask & batch_mask  # Should already be masked by batch_mask anyway
        temp_coords[~temp_mask] = 0.
        for i, (tc, m) in enumerate(zip(temp_coords, temp_mask)):
            temp_coords[i][m] -= tc[m].mean(-2)

        input.sequences = tokens
        input.template_coords = temp_coords
        input.template_mask = temp_mask
        input.batch_mask = batch_mask
        input.align_mask = align_mask

        batch_size = tokens[0].shape[0]
        seq_lens = [max(t.shape[1] - 2, 0) for t in tokens]
        seq_len = sum(seq_lens)

        return input, batch_size, seq_lens, seq_len

    def forward(
        self,
        input: IgFoldInput,
    ):
        input, batch_size, seq_lens, seq_len = self.clean_input(input)
        tokens = input.sequences
        temp_coords = input.template_coords
        temp_mask = input.template_mask
        coords_label = input.coords_label
        batch_mask = input.batch_mask
        align_mask = input.align_mask
        return_embeddings = input.return_embeddings

        res_batch_mask = rearrange(
            batch_mask,
            "b (l a) -> b l a",
            a=4,
        ).all(-1).to(self.device)
        res_temp_mask = rearrange(
            temp_mask,
            "b (l a) -> b l a",
            a=4,
        ).all(-1).to(self.device)

        ### Model forward pass

        bert_feats, bert_attns, bert_hidden = [], [], []
        for t in tokens:
            f, a, h = self.get_bert_feats(t)
            bert_feats.append(f)
            bert_attns.append(a)
            bert_hidden.append(h)

        bert_feats = torch.cat(bert_feats, dim=1)
        bert_attn = torch.zeros(
            (batch_size, seq_len, seq_len, self.bert_attn_dim),
            device=self.device,
        )
        for i, (a, l) in enumerate(zip(bert_attns, seq_lens)):
            cum_l = sum(seq_lens[:i])
            bert_attn[:, cum_l:cum_l + l, cum_l:cum_l + l, :] = a

        temp_translations, temp_rotations = self.get_coords_tran_rot(
            temp_coords,
            batch_size,
            seq_len,
        )

        str_nodes = self.str_node_transform(bert_feats)
        str_edges = self.str_edge_transform(bert_attn)
        str_nodes, str_edges = self.main_block(
            str_nodes,
            str_edges,
            mask=res_batch_mask,
        )
        gt_embs = str_nodes
        str_nodes = self.template_ipa(
            str_nodes,
            translations=temp_translations,
            rotations=temp_rotations,
            pairwise_repr=str_edges,
            mask=res_temp_mask,
        )
        structure_embs = str_nodes

        ipa_coords, ipa_translations, ipa_quaternions = self.structure_ipa(
            str_nodes,
            translations=None,
            quaternions=None,
            pairwise_repr=str_edges,
            mask=res_batch_mask,
        )
        ipa_rotations = quaternion_to_matrix(ipa_quaternions)

        dev_nodes = self.dev_node_transform(bert_feats)
        dev_edges = self.dev_edge_transform(bert_attn)
        dev_out_feats = self.dev_ipa(
            dev_nodes,
            translations=ipa_translations.detach(),
            rotations=ipa_rotations.detach(),
            pairwise_repr=dev_edges,
            mask=res_batch_mask,
        )
        dev_pred = F.relu(self.dev_linear(dev_out_feats))
        dev_pred = rearrange(dev_pred, "b l a -> b (l a)", a=4)

        bb_coords = rearrange(
            ipa_coords[:, :, :3],
            "b l a d -> b (l a) d",
        )
        flat_coords = rearrange(
            ipa_coords[:, :, :4],
            "b l a d -> b (l a) d",
        )

        ### Calculate losses if given labels
        loss = torch.zeros(
            batch_size,
            device=self.device,
        )
        if exists(coords_label):
            rmsd_clamp = self.hparams.config["rmsd_clamp"]
            coords_loss = kabsch_mse(
                flat_coords,
                coords_label,
                align_mask=batch_mask,
                mask=batch_mask,
                clamp=rmsd_clamp,
            )

            bb_coords_label = rearrange(
                rearrange(coords_label, "b (l a) d -> b l a d", a=4)[:, :, :3],
                "b l a d -> b (l a) d")
            bb_batch_mask = rearrange(
                rearrange(batch_mask, "b (l a) -> b l a", a=4)[:, :, :3],
                "b l a -> b (l a)")
            bondlen_loss = bond_length_l1(
                bb_coords,
                bb_coords_label,
                bb_batch_mask,
            )

            prmsd_loss = []
            cum_seq_lens = np.cumsum([0] + seq_lens)
            for sl_i, sl in enumerate(seq_lens):
                align_mask_ = align_mask.clone()
                align_mask_[:, :cum_seq_lens[sl_i]] = False
                align_mask_[:, cum_seq_lens[sl_i + 1]:] = False
                res_batch_mask_ = res_batch_mask.clone()
                res_batch_mask_[:, :cum_seq_lens[sl_i]] = False
                res_batch_mask_[:, cum_seq_lens[sl_i + 1]:] = False

                if sl == 0 or align_mask_.sum() == 0 or res_batch_mask_.sum(
                ) == 0:
                    continue

                prmsd_loss.append(
                    bb_prmsd_l1(
                        dev_pred,
                        flat_coords.detach(),
                        coords_label,
                        align_mask=align_mask_,
                        mask=res_batch_mask_,
                    ))
            prmsd_loss = sum(prmsd_loss)

            coords_loss, bondlen_loss = list(
                map(
                    lambda l: rearrange(l, "(c b) -> b c", b=batch_size).mean(
                        1),
                    [coords_loss, bondlen_loss],
                ))

            loss += sum([coords_loss, bondlen_loss, prmsd_loss])
        else:
            prmsd_loss, coords_loss, bondlen_loss = None, None, None

        if not exists(coords_label):
            loss = None

        bert_hidden = bert_hidden if return_embeddings else None
        bert_embs = bert_feats if return_embeddings else None
        bert_attn = bert_attn if return_embeddings else None
        gt_embs = gt_embs if return_embeddings else None
        structure_embs = structure_embs if return_embeddings else None
        output = IgFoldOutput(
            coords=ipa_coords,
            prmsd=dev_pred,
            translations=ipa_translations,
            rotations=ipa_rotations,
            coords_loss=coords_loss,
            bondlen_loss=bondlen_loss,
            prmsd_loss=prmsd_loss,
            loss=loss,
            bert_hidden=bert_hidden,
            bert_embs=bert_embs,
            bert_attn=bert_attn,
            gt_embs=gt_embs,
            structure_embs=structure_embs,
        )

        return output
    
    def score_coords(
        self,
        input: IgFoldInput,
        output: IgFoldOutput,
    ):
        input, _, _, _ = self.clean_input(input)
        batch_mask = input.batch_mask

        res_batch_mask = rearrange(
            batch_mask,
            "b (l a) -> b l a",
            a=4,
        ).all(-1)

        str_translations, str_rotations = output.translations, output.rotations

        bert_feats = output.bert_embs
        bert_attn = output.bert_attn

        dev_nodes = self.dev_node_transform(bert_feats)
        dev_edges = self.dev_edge_transform(bert_attn)
        dev_out_feats = self.dev_ipa(
            dev_nodes,
            translations=str_translations.detach(),
            rotations=str_rotations.detach(),
            pairwise_repr=dev_edges,
            mask=res_batch_mask,
        )
        dev_pred = F.relu(self.dev_linear(dev_out_feats)).squeeze(-1)
        dev_pred = rearrange(dev_pred, "b l a -> b (l a)", a=4)

        return dev_pred

    def transform_ideal_coords(self, translations, rotations):
        b, n, d = translations.shape
        device = translations.device

        ideal_coords = get_ideal_coords().to(device)
        ideal_coords = repeat(
            ideal_coords,
            "a d -> b l a d",
            b=b,
            l=n,
        )
        points_global = torch.einsum(
            'b n a c, b n c d -> b n a d',
            ideal_coords,
            rotations,
        ) + rearrange(
            translations,
            "b l d -> b l () d",
        )

        return points_global

    def gradient_refine(
        self,
        input: IgFoldInput,
        output: IgFoldOutput,
        num_steps: int = 80,
    ):
        input_, _, seq_lens, _ = self.clean_input(input)
        batch_mask = input_.batch_mask
        res_batch_mask = rearrange(
            batch_mask,
            "b (l a) -> b l a",
            a=4,
        ).all(-1)
        translations, rotations = output.translations, output.rotations

        in_coords = self.transform_ideal_coords(translations,
                                                rotations).detach()
        in_flat_coords = rearrange(
            in_coords[:, :, :4],
            "b l a d -> b (l a) d",
        )

        with torch.enable_grad():
            translations.requires_grad = True
            rotations.requires_grad = True

            translations = nn.parameter.Parameter(translations)
            rotations = nn.parameter.Parameter(rotations)

            optimizer = torch.optim.Adam([translations, rotations], lr=2e-2)
            for _ in range(num_steps):
                optimizer.zero_grad()

                coords = self.transform_ideal_coords(translations, rotations)
                viol_loss = violation_loss(coords, seq_lens, res_batch_mask)

                flat_coords = rearrange(
                    coords[:, :, :4],
                    "b l a d -> b (l a) d",
                )
                rmsd = kabsch_mse(
                    flat_coords,
                    in_flat_coords,
                    align_mask=batch_mask,
                    mask=batch_mask,
                )

                output.translations = translations
                output.rotations = rotations

                loss = rmsd + viol_loss

                loss.backward()
                optimizer.step()

        prmsd = self.score_coords(input, output)

        coords = place_o_coords(coords)
        output.coords = coords
        output.prmsd = prmsd

        return output