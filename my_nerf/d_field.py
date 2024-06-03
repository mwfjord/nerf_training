"""
Template Nerfstudio Field

Currently this subclasses the NerfactoField. Consider subclassing the base Field.
"""

from typing import Dict, Optional, Tuple, Type
import sys
try:
    import tinycudann as tcnn
except ImportError:
    pass
except EnvironmentError as _exp:
    if "Unknown compute capability" not in _exp.args[0]:
        raise _exp
    print("Could not load tinycudann: " + str(_exp), file=sys.stderr)


import torch
from torch import Tensor, nn

from jaxtyping import Float, Shaped
from nerfstudio.cameras.rays import RaySamples

from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field  # for custom Field
from nerfstudio.field_components.field_heads import DensityFieldHead, FieldHead, FieldHeadNames, RGBFieldHead
from nerfstudio.field_components.encodings import Encoding, Identity
from my_nerf.my_field import VField

class DeformationNetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(DeformationNetwork, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        modules = []
        self.mlp = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(),
                                nn.Linear(256, 256), nn.ReLU(),
                                nn.Linear(256, 256), nn.ReLU(),
                                nn.Linear(256, 256), nn.ReLU(),  
                                )
        self.mlp_skip_connection = nn.Sequential(nn.Linear(256 + in_dim, 256), nn.ReLU(),
                                nn.Linear(256, 256), nn.ReLU(),
                                nn.Linear(256, 256), nn.ReLU(),
                                nn.Linear(256, out_dim)
                                )
        modules.append(self.mlp)
        modules.append(self.mlp_skip_connection)
        self.net = nn.ModuleList(modules)
        
    def forward(self, x, t):
        # Check if t is zero tensor
        if torch.sum(t) != 0:
            mlp_in = torch.cat([x, t], dim=-1)
            # print("mlp_in", mlp_in.shape)
            h = self.mlp(mlp_in)
            mlp_out = self.mlp_skip_connection(torch.cat([h, mlp_in], dim=-1))
            return mlp_out
        else:
            return torch.zeros(x.shape[0], self.out_dim) # x.shape[0] is batch size


class DField(Field):
    
    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
        time_encoding: Encoding = Identity(in_dim=1)
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.time_encoding = time_encoding
        
        self.def_net = DeformationNetwork(
            in_dim=position_encoding.get_out_dim() + time_encoding.get_out_dim(), 
            out_dim=position_encoding.get_out_dim()
            )
        self.base_mlp_num_layers = 8
        self.base_mlp_layer_width = 256
        self.head_mlp_num_layers = 2
        self.head_mlp_layer_width = 128
        modules = []        
        self.mlp_base_1 = nn.Sequential(nn.Linear(self.position_encoding.get_out_dim(), self.base_mlp_layer_width), nn.ReLU(),
                                        nn.Linear(self.base_mlp_layer_width, self.base_mlp_layer_width), nn.ReLU(),
                                        nn.Linear(self.base_mlp_layer_width, self.base_mlp_layer_width), nn.ReLU(),
                                        nn.Linear(self.base_mlp_layer_width, self.base_mlp_layer_width), nn.ReLU()
                                      )
        self.mlp_base_2_skip_connection = nn.Sequential(nn.Linear(self.base_mlp_layer_width + self.position_encoding.get_out_dim(), self.base_mlp_layer_width), nn.ReLU(),
                                        nn.Linear(self.base_mlp_layer_width, self.base_mlp_layer_width), nn.ReLU(),
                                        nn.Linear(self.base_mlp_layer_width, self.base_mlp_layer_width), nn.ReLU(),
                                        nn.Linear(self.base_mlp_layer_width, self.base_mlp_layer_width)
                                        )
        self.field_output_density = DensityFieldHead(in_dim=self.base_mlp_layer_width)
        self.mlp_head = nn.Sequential(nn.Linear(self.direction_encoding.get_out_dim() + self.base_mlp_layer_width, self.head_mlp_layer_width), nn.ReLU(),
                                     nn.Linear(self.head_mlp_layer_width, self.head_mlp_layer_width), nn.Sigmoid()
                                    )
        modules.append(self.def_net)
        modules.append(self.mlp_base_1)
        modules.append(self.mlp_base_2_skip_connection) 
        modules.append(self.mlp_head)
        self.modules = nn.ModuleList(modules)
        self.field_output_color = RGBFieldHead(in_dim=self.head_mlp_layer_width)
        
    # or subclass from base Field and define all mandatory methods.
    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        positions = ray_samples.frustums.get_positions()
        times = ray_samples.times
        encoded_xyz = self.position_encoding(positions)
        time_encoding = self.time_encoding(times)
        deform_out = self.def_net(encoded_xyz, time_encoding)
        # print("deform_out", deform_out.shape)
        deform_out = torch.add(deform_out, encoded_xyz)
        # print("deform_out", deform_out.shape)

        base_mlp_1_out = self.mlp_base_1(deform_out)
        # print("base_mlp_1_out", base_mlp_1_out.shape)
        base_mlp_out = self.mlp_base_2_skip_connection(torch.cat([base_mlp_1_out, encoded_xyz], dim=-1))
        density =  self.field_output_density(base_mlp_out)
        return density, base_mlp_out
        
    
    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:

        encoded_dir = self.direction_encoding(ray_samples.frustums.directions)
        mlp_out = self.mlp_head(torch.cat([encoded_dir, density_embedding], dim=-1))
        mlp_out = self.field_output_color(mlp_out)
        outputs = {FieldHeadNames.RGB: mlp_out}
        return outputs
