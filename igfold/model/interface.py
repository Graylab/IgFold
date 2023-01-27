from dataclasses import dataclass
from typing import List, Optional, Union
import torch


@dataclass
class IgFoldInput():
    """
    Input type of for IgFold model.
    """

    sequences: List[Union[torch.LongTensor, str]]
    template_coords: Optional[torch.FloatTensor] = None
    template_mask: Optional[torch.BoolTensor] = None
    batch_mask: Optional[torch.BoolTensor] = None
    align_mask: Optional[torch.BoolTensor] = None
    coords_label: Optional[torch.FloatTensor] = None
    return_embeddings: Optional[bool] = False


@dataclass
class IgFoldOutput():
    """
    Output type of for IgFold model.
    """

    coords: torch.FloatTensor
    prmsd: torch.FloatTensor
    translations: torch.FloatTensor
    rotations: torch.FloatTensor
    coords_loss: Optional[torch.FloatTensor] = None
    torsion_loss: Optional[torch.FloatTensor] = None
    bondlen_loss: Optional[torch.FloatTensor] = None
    prmsd_loss: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    bert_hidden: Optional[torch.FloatTensor] = None
    bert_embs: Optional[torch.FloatTensor] = None
    bert_attn: Optional[torch.FloatTensor] = None
    gt_embs: Optional[torch.FloatTensor] = None
    structure_embs: Optional[torch.FloatTensor] = None