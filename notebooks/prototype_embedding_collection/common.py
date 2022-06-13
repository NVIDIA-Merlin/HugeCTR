import typing


class EmbeddingParam(typing.NamedTuple):
    embedding_id: int
    id_space: int
    combiner: str  # concat, sum, mean
    hotness: int
    ev_size: int

class EmbeddingCollectionParam(typing.NamedTuple):
    num_embeddings: int
    embedding_params: typing.List[EmbeddingParam]
    universal_batch_size: int
    key_type : str
    offset_type: str
    emb_type: str
    
class EmbeddingShardingParam(typing.NamedTuple):
    local_embedding_list: typing.List[int]
    global_embedding_list: typing.List[typing.List[int]]
    sharding_id: int
    num_sharding: int
    table_placement_strategy: str  # dp, localized

def has_concat_embedding(embedding_params: typing.List[EmbeddingParam]) -> bool:
    return any([b.combiner == "concat" for b in embedding_params])
