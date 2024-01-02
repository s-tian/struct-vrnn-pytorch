import numpy as np
import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self, num_input_channels=5, input_map_size=64, output_map_size=16, initial_num_filters=32,
                 layers_per_scale=1, use_batchnorm=False, **conv_layers_kwargs):
        super().__init__()
        self.num_input_channels = num_input_channels  # Default RGB=3 + 2 for coordinate channels
        self.input_map_size = input_map_size
        self.initial_num_filters = initial_num_filters
        self.output_map_size = output_map_size
        self.layers_per_scale = layers_per_scale
        self.conv_layers_kwargs = conv_layers_kwargs

        # Make sure that input_map_size / output_map_width is a power of 2
        assert input_map_size / output_map_size == 2 ** int(np.log2(input_map_size / output_map_size)), \
            "input_map_size / output_map_size must be a power of 2"

        self.layers = nn.ModuleList()

        # Layers that increase to initial number of filters
        self.layers.append(nn.Conv2d(num_input_channels, initial_num_filters, kernel_size=3, padding=1))
        for _ in range(self.layers_per_scale):
            self.layers.append(nn.Conv2d(initial_num_filters, initial_num_filters, kernel_size=3, padding=1))

        # Downsampling blocks
        num_filters = initial_num_filters
        for downsample_factor in range(int(np.log2(input_map_size / output_map_size))):
            num_filters *= 2
            self.layers.append(nn.Conv2d(num_filters // 2, num_filters, stride=2, kernel_size=3, padding=1))
            for _ in range(self.layers_per_scale):
                if use_batchnorm:
                    self.layers.append(nn.BatchNorm2d(num_filters))
                self.layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1))

        self.nonlinearity = nn.LeakyReLU(0.2)

    @property
    def output_num_filters(self):
        return self.initial_num_filters * 2 ** int(np.log2(self.input_map_size / self.output_map_size))

    def forward(self, images):
        # Image encoder forward pass. Does not compute the softplus activation on the output.
        x = images
        for layer in self.layers:
            x = layer(x)
            x = self.nonlinearity(x)
        return x


class ImageDecoder(nn.Module):

    def __init__(self, num_input_channels=15, input_map_size=16, output_map_size=64, initial_num_filters=128,
                 layers_per_scale=1, use_batchnorm=False, **conv_layers_kwargs):
        super().__init__()
        self.num_input_channels = num_input_channels
        self.input_map_size = input_map_size
        # initial num filters should be the number of output channels of the image encoder
        self.initial_num_filters = initial_num_filters
        self.output_map_size = output_map_size
        self.layers_per_scale = layers_per_scale
        self.conv_layers_kwargs = conv_layers_kwargs

        # Make sure that input_map_size / output_map_width is a power of 2
        assert output_map_size / input_map_size == 2 ** int(np.log2(output_map_size / input_map_size)), \
            "output_map_size / input_map_size must be a power of 2"

        self.layers = nn.ModuleList()

        # Layers that increase to initial number of filters
        self.layers.append(nn.Conv2d(num_input_channels, initial_num_filters, kernel_size=1, padding=0))

        # Upsampling blocks
        num_filters = initial_num_filters
        for upsample_factor in range(int(np.log2(output_map_size / input_map_size))):
            num_filters //= 2
            self.layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
            for layer_idx in range(self.layers_per_scale):
                if layer_idx == 0:
                    # The first layer in each upsampling block needs to scale down the number of filters
                    if use_batchnorm:
                        self.layers.append(nn.BatchNorm2d(num_filters*2))
                    self.layers.append(nn.Conv2d(num_filters * 2, num_filters, kernel_size=3, padding=1))
                else:
                    if use_batchnorm:
                        print("Adding BN")
                        self.layers.append(nn.BatchNorm2d(num_filters))
                    self.layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1))
        # adjust channels of output image layer
        self.layers.append(nn.Conv2d(num_filters, 3, kernel_size=1, padding=0))
        self.nonlinearity = nn.LeakyReLU(0.2)

    def forward(self, images):
        # Image encoder forward pass. Does not compute the softplus activation on the output.
        x = images
        for layer_num, layer in enumerate(self.layers):
            x = layer(x)
            if isinstance(layer, nn.Conv2d) and layer_num != len(self.layers) - 1:
                x = self.nonlinearity(x)
        return x


class CoordChannelLayer(nn.Module):

    def __init__(self, input_map_size=16):
        super().__init__()
        self.input_map_size = input_map_size
        self.register_buffer("x_coords", torch.linspace(-1, 1, input_map_size))
        self.register_buffer("y_coords", torch.linspace(1, -1, input_map_size))

    def forward(self, input):
        # Add coordinate channels to the input
        input_dims = input.shape[:-3]
        final_shape = input_dims + (1, self.input_map_size, self.input_map_size)
        x_coords = self.x_coords.expand(final_shape)
        y_coords = self.y_coords[:, None].expand(final_shape)
        return torch.cat([input, x_coords, y_coords], dim=-3)


class KeypointDetector(nn.Module):

    EPS = 1e-6

    def __init__(self, encoder, appearance_encoder, decoder, num_keypoints, keypoint_sigma):
        super().__init__()
        self.image_encoder = encoder
        self.encoder_coord_channel_layer = CoordChannelLayer(encoder.input_map_size)
        self.decoder_coord_channel_layer = CoordChannelLayer(encoder.output_map_size)
        self.encoded_image_to_heatmaps = nn.Conv2d(self.image_encoder.output_num_filters, num_keypoints, kernel_size=1, padding=0)

        # Create coordinate axes for heatmap space and register as buffers so they move to the correct devices
        x_coord_axis = torch.linspace(-1, 1, encoder.output_map_size)
        y_coord_axis = torch.linspace(1, -1, encoder.output_map_size)
        self.register_buffer("x_coord_axis", x_coord_axis)
        self.register_buffer("y_coord_axis", y_coord_axis)

        self.image_decoder = decoder
        self.appearance_encoder = appearance_encoder
        self.num_keypoints = num_keypoints
        self.sigma = keypoint_sigma

    def heatmaps_to_keypoints(self, heatmaps):
        # Convert heatmaps to keypoints
        # heatmaps: (batch_size, num_keypoints, map_size, map_size)
        # keypoints: (batch_size, num_keypoints, 3)
        intensities = torch.mean(heatmaps, dim=(2, 3))
        # Normalize intensities across keypoints.
        intensities = intensities / (torch.amax(intensities, dim=-1, keepdim=True) + self.EPS)
        normalized_heatmaps = heatmaps / (heatmaps.sum(dim=(2, 3), keepdim=True) + self.EPS)
        # TODO: Double check that the marginalization is correctly performed
        x_coords = (normalized_heatmaps * self.x_coord_axis).sum(dim=(2, 3))
        y_coords = (normalized_heatmaps * self.y_coord_axis[..., None]).sum(dim=(2, 3))
        return torch.stack([x_coords, y_coords, intensities], dim=-1)

    def image_to_keypoints(self, images):
        images_with_coords = self.encoder_coord_channel_layer(images)
        image_features = self.image_encoder(images_with_coords)
        heatmaps = self.encoded_image_to_heatmaps(image_features)
        # Heatmaps need to be non-negative
        heatmaps = nn.functional.softplus(heatmaps)
        keypoints = self.heatmaps_to_keypoints(heatmaps)
        return keypoints

    def keypoints_to_gaussian_map(self, keypoints):
        x_coords = keypoints[..., 0]
        y_coords = keypoints[..., 1]
        intensity = keypoints[..., 2]
        keypoint_width = 2.0 * (self.sigma / self.image_encoder.output_map_size) ** 2
        # Compute marginals for each keypoint
        x_vec = torch.exp(-torch.square(self.x_coord_axis - x_coords[..., None, None]) / keypoint_width)
        y_vec = torch.exp(-torch.square(self.y_coord_axis[..., None] - y_coords[..., None, None]) / keypoint_width)
        # Multiply marginals to get joint dist.
        gaussian_maps = x_vec * y_vec
        # Scale Gaussian maps based on intensity
        gaussian_maps = gaussian_maps * intensity[..., None, None]
        return gaussian_maps

    def keypoints_to_image(self, keypoints, first_frame, first_frame_keypoints):
        gaussian_maps = self.keypoints_to_gaussian_map(keypoints)
        first_frame_features = self.appearance_encoder(first_frame)
        first_frame_keypoint_maps = self.keypoints_to_gaussian_map(first_frame_keypoints)
        decoder_inputs = torch.cat([gaussian_maps, first_frame_features, first_frame_keypoint_maps], dim=1)
        decoder_inputs = self.decoder_coord_channel_layer(decoder_inputs)
        reconstructed_image = self.image_decoder(decoder_inputs) + first_frame
        # reconstructed_image = self.image_decoder(decoder_inputs)
        return reconstructed_image

    def test_visualize_keypoints_to_gaussian_map(self):
        keypoints = torch.zeros((1, self.num_keypoints, 3)).cuda()
        keypoints[0, 0] = torch.tensor([0.5, 0.5, 1.0]).cuda()
        keypoints[0, 1] = torch.tensor([-0.5, -0.5, 1.0]).cuda()
        keypoints[0, 2] = torch.tensor([0, 0, 1.0]).cuda()
        gmap = self.keypoints_to_gaussian_map(keypoints)
        # get first three channels of gmap
        gmap = gmap[0, :3]
        # write it as an image to disk
        gmap = gmap.permute(1, 2, 0)
        gmap = gmap.detach().cpu().numpy()
        gmap = (gmap * 255).astype(np.uint8)
        import cv2
        cv2.imwrite("gmap.png", gmap)
        print(f"Writing debug keypoint visualization to gmap.png...")
