"""PA-TCNet built with the paper's settings, from the original model code."""
from .original.model import PATCNet
from .settings import DATASETS, DatasetSettings, ModelSettings, flattened_features


def build_model(dataset: str, model: ModelSettings = ModelSettings(), settings: DatasetSettings | None = None) -> PATCNet:
    settings = settings or DATASETS[dataset]
    return PATCNet(embedding_dim=model.embedding, depth=model.depth,
                   temporal_filters_per_branch=model.temporal_filters_per_branch,
                   spatial_multiplier=model.spatial_multiplier, first_pool_size=model.pool,
                   second_pool_size=model.pool, dropout_rate=model.dropout, num_channels=settings.channels,
                   num_classes=2, flattened_feature_dim=flattened_features(settings.samples, model),
                   temporal_kernel_sizes=model.temporal_kernels)
