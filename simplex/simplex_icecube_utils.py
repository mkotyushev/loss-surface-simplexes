from simplex.models.simplex_models import Linear
from typing import Union
from torch_geometric.typing import Adj
from torch.functional import Tensor
from torch_geometric.typing import Adj, PairTensor
import mock

# replace linear layer so that it can accept the coeffs_t 
# in forward method

class LinearGraphnet(Linear):
    def forward(self, input):
        input, coeffs_t = input
        return super().forward(input, coeffs_t), coeffs_t

# mock callstack until each linear layer to forward the coeffs_t
# to linear layer forward method

class EdgeConvMock:
    @mock.patch("torch_geometric.nn.conv.edge_conv.EdgeConv.forward")
    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

        return x, edge_index

def task_forward():
    pass
