import abc
from embedding_table import *

class IOptimizer(abc.ABC):
    @abc.abstractmethod
    def update(self, unique_key, num_unique_key, unique_key_id_space_list, unique_key_is_space_offset, grad_ev, grad_ev_id_space_offset, embedding_table):
        pass


class SGDOptimizer(IOptimizer):
    def __init__(self,  param_list: typing.List[EmbeddingTableParam]) -> None:
        super().__init__()
        if len(param_list) > 0:
            assert any([p.vocabulary_size != -1 for p in param_list])  # do not support dynamic size
            self.id_space_list = [p.id_space for p in param_list]
            self.id_space_idx = [-1 for _ in range(max(self.id_space_list) + 1)]
            for idx, id_space in enumerate(self.id_space_list):
                self.id_space_idx[id_space] = idx

            self.ev_size_list = [p.ev_size for p in param_list]
            self.max_vocabulary_size = sum([p.vocabulary_size for p in param_list])
        else:
            self.id_space_list = []
            self.id_space_idx = []

            self.ev_size_list = []
            self.max_vocabulary_size = 0

    def update(self, unique_key, num_unique_key, unique_key_id_space_list, unique_key_is_space_offset, grad_ev, grad_ev_id_space_offset, embedding_table):
        pass
