from thunder.models import PretrainedModel


class Resnet18Features(PretrainedModel):
    def __init__(self):
        super().__init__()

        import torch
        from torchvision import transforms
        from torchvision.models import resnet18

        resnet = resnet18()

        self.name = "resnet18"
        self.emb_dim = 512
        self.vlm = False

        self.feature_maps = torch.nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.features = torch.nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(start_dim=1),
        )

        self.t = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def forward(self, x):
        x = self.features(x)
        return x

    def get_transform(self):
        return self.t

    def get_linear_probing_embeddings(self, x):
        x = self.features(x)
        return x

    def get_segmentation_embeddings(self, x):
        x = self.feature_maps(x)
        x = x.view(x.shape[0], x.shape[1], -1).swapdims(1, 2)
        return x
