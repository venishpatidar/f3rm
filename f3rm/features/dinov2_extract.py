import gc
from typing import List

import torch
from einops import rearrange
from tqdm import tqdm


class DINOV2Args:
    model_type: str = "dinov2_vits14"
    load_size: int = 224
    stride: int = 7
    facet: str = "key"
    layer: int = 11
    bin: bool = False
    batch_size: int = 4

    @classmethod
    def id_dict(cls):
        """Return dict that identifies the DINO model parameters."""
        return {
            "model_type": cls.model_type,
            "load_size": cls.load_size,
            "stride": cls.stride,
            "facet": cls.facet,
            "layer": cls.layer,
            "bin": cls.bin,
        }


_supported_dinov2_models = {"dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"}


@torch.no_grad()
def extract_dinov2_features(image_paths: List[str], device: torch.device) -> torch.Tensor:
    from f3rm.features.dinov2.dinov2_vit_extractor import ViTExtractor

    assert (
        DINOV2Args.model_type in _supported_dinov2_models
    ), f"Model type must be one of {_supported_dinov2_models}, not {DINOV2Args.model_type}"

    extractor = ViTExtractor(DINOV2Args.model_type, DINOV2Args.stride, device=device)
    print(f"Loaded DINOV2 model {DINOV2Args.model_type}")
    # Preprocess images
    preprocessed_images = [extractor.preprocess(image_path, DINOV2Args.load_size)[0] for image_path in image_paths]
    preprocessed_images = torch.cat(preprocessed_images, dim=0).to(device)
    print(f"Preprocessed {len(image_paths)} images to shape {preprocessed_images.shape}")

    # Extract DINOV2 features in batches
    embeddings = []
    for i in tqdm(
        range(0, len(preprocessed_images), DINOV2Args.batch_size),
        desc="Extracting DINOV2 features",
    ):
        batch = preprocessed_images[i : i + DINOV2Args.batch_size]
        embeddings.append(extractor.extract_descriptors(batch, DINOV2Args.layer, DINOV2Args.facet, DINOV2Args.bin))
    embeddings = torch.cat(embeddings, dim=0)

    # Reshape embeddings to have shape (batch, height, width, channels))
    height, width = extractor.num_patches
    embeddings = rearrange(embeddings, "b 1 (h w) c -> b h w c", h=height, w=width)
    print(f"Extracted DINOV2 embeddings of shape {embeddings.shape}")

    # Delete and clear memory to be safe
    del extractor
    del preprocessed_images
    torch.cuda.empty_cache()
    gc.collect()

    return embeddings

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    import os
    _MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
    _IMAGE_DIR = os.path.join(_MODULE_DIR, "images")

    image_paths = [os.path.join(_IMAGE_DIR, name) for name in ["frame_1.png", "frame_2.png", "frame_3.png"]]

    dinov2_embeddings = extract_dinov2_features(image_paths, device)
    print(dinov2_embeddings)
