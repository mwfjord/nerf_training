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



class VField(Field):
    
    def __init__(
        self,
        position_encoding: Encoding = Identity(in_dim=3),
        direction_encoding: Encoding = Identity(in_dim=3),
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.base_mlp_num_layers = 8
        self.base_mlp_layer_width = 256
        self.head_mlp_num_layers = 2
        self.head_mlp_layer_width = 128
        self.relu = nn.ReLU()
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
        modules.append(self.mlp_base_1)
        modules.append(self.mlp_base_2_skip_connection) 
        modules.append(self.mlp_head)
        self.modules = nn.ModuleList(modules)
        self.field_output_color = RGBFieldHead(in_dim=self.head_mlp_layer_width)
                
        
        
    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        positions = ray_samples.frustums.get_positions()
        encoded_xyz = self.position_encoding(positions)
        base_mlp_1_out = self.mlp_base_1(encoded_xyz)
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
