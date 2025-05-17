import torch
import torch.nn as nn

class Model3_ViT(nn.Module):
    def __init__(self, backbone, config,
                 input_dim=2048, num_tokens=40, num_classes=30, d_model=512, nhead=8, num_layers=2):
        super(Model3_ViT, self).__init__()
        self.name = "resNet50 + ViT"
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(1024, 30)
        self.config = config
        
        
        # Linear layer to project input to the desired embedding dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        # Positional encoding for the sequence length (num_tokens)
        self.positional_encoding = nn.Parameter(torch.randn(1, num_tokens, d_model))
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, **kwargs):
        batch = dict(**kwargs)
        # x: (B, N_IMAGES, C, H, W)
        features_output = []
        for series_description in ['Sagittal T2/STIR', 'Sagittal T1', 'Axial T2']:
            if not series_description in batch:
                # not used, we just forward zeros through backbone
                features_output.append(None)
            else:
                gray_images = batch[series_description].to('cuda')
                images = gray_images.repeat(1, 1, 3, 1, 1)
                
                # Normalize images
                mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 1, 3, 1, 1)
                images = (images - mean) / std
                
                B, N_IMAGES, C, H, W = images.shape
                if self.config.model.train_backbone:
                    features = self.backbone.forward_features(images.view(B * N_IMAGES, C, H, W))
                else:
                    with torch.no_grad():
                        features = self.backbone.forward_features(images.view(B * N_IMAGES, C, H, W))
                features = self.pool(features).squeeze(-1).squeeze(-1).view(B, N_IMAGES, -1)
                
                features_output.append(features)
        for i in range(len(features_output)):
            if features_output[i] is None:
                features_output[i] = torch.zeros((B, N_IMAGES, self.backbone.num_features), device=images.device)
        features_output = torch.cat(features_output, 1)
        
        
        # Project input to d_model dimension
        x = self.input_projection(features_output)  # Shape: (B, 40, d_model)
        # Add positional encoding
        x = x + self.positional_encoding  # Shape: (B, 40, d_model)
        # Permute to (seq_len, batch, embedding_dim) for transformer
        x = x.permute(1, 0, 2)  # Shape: (40, B, d_model)
        # Transformer Encoder
        x = self.transformer_encoder(x)  # Shape: (40, B, d_model)
        # Take the mean of the sequence dimension for classification
        x = x.mean(dim=0)  # Shape: (B, d_model
        # Classification head
        output = self.classifier(x)  # Shape: (B, num_classes)
        
#         output = self.classifier(features_output)
        output = output.view(-1, 10, 3)
            
        loss = None
        if 'labels' in batch:
            if self.config.model.use_weights:
                loss_fn = torch.nn.CrossEntropyLoss(torch.tensor([1., 2., 4.], device=output.device))
            else:
                loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(output.view(-1, 3), batch['labels'].view(-1, ).to("cuda"))
        
        return loss, output