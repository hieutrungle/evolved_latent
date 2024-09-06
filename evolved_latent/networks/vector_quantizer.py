import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, name=None):
        """Return a vector quantized data based on distance criteria.
        Quantization with one hot encoding and a learnable Embedding table
        """
        super().__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self.name = name

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape_channel_last = inputs.shape

        # Flatten input, (B, H, W, C) -> (B * H * W, C)
        flat_input = inputs.view(-1, self._embedding_dim)

        # (B * H * W, 1)
        encoding_indices = self.get_code_indices(flat_input)
        # Encoding, one hot encoding based on the encoding_indices
        # (B * H * W, 1) -> (B * H * W, num_embeddings)
        encodings = self.get_encodings(encoding_indices)

        # Quantize (B * H * W, num_embeddings) * (num_embeddings, C) -> (B * H * W, C)
        # Quantization example:
        # encodings= [[0,1,0], [1,0,0]]; self._embedding.weight = [[1,2,3], [4,5,6], [7,8,9]]
        # When we do matmul, we get the second row and the first row of self._embedding.weight
        # Results: [[4,5,6], [1,2,3]]
        quantized = torch.matmul(encodings, self._embedding.weight)
        # Unflatten quantized, (B * H * W, C) -> (B, H, W, C)
        quantized = quantized.view(input_shape_channel_last)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

    def get_code_indices(self, flat_input):
        # Calculate distances, (B * H * W, num_embeddings)
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Get the smallest distance for each element in dim=0 along the dim=1
        # (B * H * W, num_embeddings) -> (B * H * W, 1)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        return encoding_indices

    def get_encodings(self, encoding_indices):
        # one hot encoding based on the encoding_indices
        # (B * H * W, 1) -> (B * H * W, num_embeddings)
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self._num_embeddings,
            device=encoding_indices.device,
        )
        encodings.scatter_(dim=1, index=encoding_indices, value=1)
        return encodings

    def get_quantized_from_indices(
        self, encoding_indices, quantized_shape_channel_last
    ):
        # (B * H * W, 1) -> (B * H * W, num_embeddings)
        encodings = self.get_encodings(encoding_indices)
        # (BxHxW, num_embeddings) * (embedding_dim, num_embeddings).T = (BxHxW, embedding_dim)
        quantized = torch.matmul(encodings, self._embedding.weight)
        # Unflatten quantized, (B * H * W, C) -> (B, H, W, C)
        quantized = torch.reshape(
            quantized,
            (
                -1,
                quantized_shape_channel_last[-3],
                quantized_shape_channel_last[-2],
                quantized_shape_channel_last[-1],
            ),
        )
        # (B, C, H, W)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized


class VectorQuantizerEMA(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
        name=None,
    ):
        super().__init__()

        self.name = name
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape_channel_last = inputs.shape

        # Flatten input, (B, H, W, C) -> (B * H * W, C)
        flat_input = inputs.view(-1, self._embedding_dim)

        # (B * H * W, 1)
        encoding_indices = self.get_code_indices(flat_input)
        # (B * H * W, 1) -> (B * H * W, num_embeddings)
        encodings = self.get_encodings(encoding_indices)

        # Quantize (B * H * W, num_embeddings) * (num_embeddings, C) -> (B * H * W, C)
        # Quantization example:
        # encodings= [[0,1,0], [1,0,0]]; self._embedding.weight = [[1,2,3], [4,5,6], [7,8,9]]
        # When we do matmul, we get the second row and the first row of self._embedding.weight
        # Results: [[4,5,6], [1,2,3]]
        quantized = torch.matmul(encodings, self._embedding.weight)
        # Unflatten quantized, (B * H * W, C) -> (B, H, W, C)
        quantized = quantized.view(input_shape_channel_last)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

    def get_code_indices(self, flat_input):
        # Calculate distances, (B * H * W, num_embeddings)
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Get the smallest distance for each element in dim=0 along the dim=1
        # (B * H * W, num_embeddings) -> (B * H * W, 1)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        return encoding_indices

    def get_encodings(self, encoding_indices):
        # one hot encoding based on the encoding_indices
        # (B * H * W, 1) -> (B * H * W, num_embeddings)
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self._num_embeddings,
            device=encoding_indices.device,
        )
        # encodings.scatter_(dim=1, index=encoding_indices, value=1)
        # encodings_1 = F.one_hot(
        #     encoding_indices[:, 0], num_classes=self._num_embeddings
        # )
        # encodings_1 = encodings_1.type(self._embedding.weight.dtype)

        encodings[torch.arange(encodings.shape[0]), encoding_indices[:, 0]] = 1.0
        encodings = encodings.type(self._embedding.weight.dtype)
        return encodings

    def get_quantized_from_indices(
        self, encoding_indices, quantized_shape_channel_last
    ):
        # (B * H * W, 1) -> (B * H * W, num_embeddings)
        encodings = self.get_encodings(encoding_indices)
        # (BxHxW, num_embeddings) * (embedding_dim, num_embeddings).T = (BxHxW, embedding_dim)
        quantized = torch.matmul(encodings, self._embedding.weight)
        # Unflatten quantized, (B * H * W, C) -> (B, H, W, C)
        quantized = torch.reshape(
            quantized,
            (
                -1,
                quantized_shape_channel_last[-3],
                quantized_shape_channel_last[-2],
                quantized_shape_channel_last[-1],
            ),
        )
        # (B, C, H, W)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized


class VectorQuantizer1DEMA(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost=0.25,
        decay=0.99,
        epsilon=1e-5,
        name=None,
    ):
        super().__init__()

        self.name = name
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer("_ema_cluster_size", torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCL -> BLC
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape_channel_last = inputs.shape

        # Flatten input, (B, L, C) -> (B * L, C)
        flat_input = inputs.view(-1, self._embedding_dim)

        # (B * L, 1)
        encoding_indices = self.get_code_indices(flat_input)
        # (B * L, 1) -> (B * L, num_embeddings)
        encodings = self.get_encodings(encoding_indices)

        # Quantize (B * L, num_embeddings) * (num_embeddings, C) -> (B * L, C)
        # Quantization example:
        # encodings= [[0,1,0], [1,0,0]]; self._embedding.weight = [[1,2,3], [4,5,6], [7,8,9]]
        # When we do matmul, we get the second row and the first row of self._embedding.weight
        # Results: [[4,5,6], [1,2,3]]
        quantized = torch.matmul(encodings, self._embedding.weight)
        # Unflatten quantized, (B * L, C) -> (B, L, C)
        quantized = quantized.view(input_shape_channel_last)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + (
                1 - self._decay
            ) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon)
                * n
            )

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw
            )

            self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1)
            )

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BLC -> BCL
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings

    def get_code_indices(self, flat_input):
        # Calculate distances, (B * L, num_embeddings)
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        # Get the smallest distance for each element in dim=0 along the dim=1
        # (B * L, num_embeddings) -> (B * L, 1)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        return encoding_indices

    def get_encodings(self, encoding_indices):
        # one hot encoding based on the encoding_indices
        # (B * L, 1) -> (B * L, num_embeddings)
        encodings = torch.zeros(
            encoding_indices.shape[0],
            self._num_embeddings,
            device=encoding_indices.device,
        )
        # encodings.scatter_(dim=1, index=encoding_indices, value=1)
        # encodings_1 = F.one_hot(
        #     encoding_indices[:, 0], num_classes=self._num_embeddings
        # )
        # encodings_1 = encodings_1.type(self._embedding.weight.dtype)

        encodings[torch.arange(encodings.shape[0]), encoding_indices[:, 0]] = 1.0
        encodings = encodings.type(self._embedding.weight.dtype)
        return encodings

    def get_quantized_from_indices(
        self, encoding_indices, quantized_shape_channel_last
    ):
        # (B * L, 1) -> (B * L, num_embeddings)
        encodings = self.get_encodings(encoding_indices)
        # (BxHxW, num_embeddings) * (embedding_dim, num_embeddings).T = (BxHxW, embedding_dim)
        quantized = torch.matmul(encodings, self._embedding.weight)
        # Unflatten quantized, (B * L, C) -> (B, H, W, C)
        quantized = torch.reshape(
            quantized,
            (
                -1,
                quantized_shape_channel_last[-3],
                quantized_shape_channel_last[-2],
                quantized_shape_channel_last[-1],
            ),
        )
        # (B, C, H, W)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return quantized
