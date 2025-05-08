import torch
from e3nn import o3
from e3nn.nn import Gate

class LearnableGate(torch.nn.Module):
    """
    Learnable gate for E3NN transformation.
    
    Fixes the issue of e3nn.nn.Gate having different sizes for input and output irreps.
    """
    def __init__(self, irreps_in):
        super(LearnableGate, self).__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        
        # Separate input irreps into scalars and non-scalars
        self.irreps_scalars = o3.Irreps([(mul, (l, p)) for mul, (l, p) in self.irreps_in if l == 0])
        self.irreps_non_scalars = o3.Irreps([(mul, (l, p)) for mul, (l, p) in self.irreps_in if l > 0])
        
        # Calculate number of scalars and non-scalars
        non_scalar_count = sum(mul for mul, (_, _) in self.irreps_non_scalars)
        self.irreps_gate_scalars = o3.Irreps(f"{non_scalar_count}x0e")
        
        # Create standard gate input irreps
        self.standard_gate_input = self.irreps_scalars + self.irreps_gate_scalars + self.irreps_non_scalars
        
        # Linear transformation for input
        self.linear = o3.Linear(
            irreps_in=self.irreps_in,
            irreps_out=self.standard_gate_input,
        )
        
        # Gate for activation
        self.gate = Gate(
            irreps_scalars=self.irreps_scalars,
            act_scalars=[torch.tanh],
            irreps_gates=self.irreps_gate_scalars,
            act_gates=[torch.tanh],
            irreps_gated=self.irreps_non_scalars
        )
    
    def forward(self, x):
        return self.gate(self.linear(x))

class TensorProductBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps
    ):
        super(TensorProductBlock, self).__init__()
        
        gate_required_irreps = irreps_out.sort().irreps.simplify()
        
        self.lin1 = o3.Linear(irreps_in=irreps_in1, irreps_out=irreps_in1, internal_weights=True)
        self.lin2 = o3.Linear(irreps_in=irreps_in2, irreps_out=irreps_in2, internal_weights=True)
        
        self.tp = o3.FullyConnectedTensorProduct(
            irreps_in1=irreps_in1,
            irreps_in2=irreps_in2,
            irreps_out=gate_required_irreps
        )
        
        self.gate = LearnableGate(gate_required_irreps)
        
        self.lin_out = o3.Linear(gate_required_irreps, irreps_out)
    
    def forward(self, x):
        x_1 = self.lin1(x)
        x_2 = self.lin2(x)
        x = self.tp(x_1, x_2)
        x = self.gate(x)
        return self.lin_out(x)

class EquivariantModel(torch.nn.Module):
    def __init__(
        self, 
        irreps_in: o3.Irreps,
        irreps_hidden: o3.Irreps,
        irreps_out: o3.Irreps,
    ) -> None:
        super(EquivariantModel, self).__init__()
        
        self.irreps_in = irreps_in
        self.irreps_hidden = irreps_hidden
        self.irreps_out = irreps_out
        
        self.tensorproduct_1 = TensorProductBlock(
            irreps_in1=irreps_in,
            irreps_in2=irreps_in,
            irreps_out=irreps_hidden
        )
        
        self.tensorproduct_2 = TensorProductBlock(
            irreps_in1=irreps_hidden,
            irreps_in2=irreps_hidden,
            irreps_out=irreps_hidden
        )
        
        self.lin_out = o3.Linear(
            irreps_in=irreps_hidden,
            irreps_out=irreps_out
        )
        
    
    def forward(self, x):
        x = self.tensorproduct_1(x)
        x = self.tensorproduct_2(x)
        x = self.lin_out(x)
        return x