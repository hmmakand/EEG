from .inductive_node import GNNInductiveNodeHead
from .san_graph import SANGraphHead

head_dict = {'san_graph': SANGraphHead,
             'inductive_node': GNNInductiveNodeHead}
