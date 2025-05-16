from thunder.models import PretrainedModel


class DINOv2Features(PretrainedModel):
    def __init__(self):
        super().__init__()

        import torch
        from torchvision import transforms

        self.dinov2 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.t = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
            ]
        )
        self.name = "dinov2_vits14"
        self.emb_dim = 384

    def forward(self, x):
        feats = self.dinov2.forward_features(x)
        return feats

    def get_transform(self):
        return self.t

    def get_linear_probing_embeddings(self, x):
        x = self.dinov2.forward_features(x)
        return x["x_norm_clstoken"]

    def get_segmentation_embeddings(self, x):
        x = self.dinov2.forward_features(x)
        return x["x_norm_patchtokens"]
