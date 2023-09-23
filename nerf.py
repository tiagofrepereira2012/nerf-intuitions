from pathlib import Path
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import argparse

from volumetric_rendering import (
    SphereVolumetricRendering,
    rendering,
    compute_transmittance,
)

from voxel_reconstruction import get_rays, training_loop


class NeRF(nn.Module):
    """
    Neural Radiance Field

    The goal is to predict the color and the density of a point in the 3D space

    Args:
        l_position: Number of layers for the positional encoding
        l_direction: Number of layers for the direction encoding
        hidden_size: Hidden size of the model
        with_pos_enc: Whether to use positional encoding or not
    """

    def __init__(self, l_position=10, l_direction=4, hidden_size=256, with_pos_enc=True):
        super(NeRF, self).__init__()

        # Instead of representing with a voxel model, we will do it with an NN
        # self.voxels = nn.Parameter(
        #    torch.rand(nb_voxels, nb_voxels, nb_voxels, 4, device=device),
        #    requires_grad=True,
        # )

        self.with_pos_enc = with_pos_enc

        # Why  *2 (sinus and cosinus for the positional encoding) and *3 for the x,y,z coordinates
        # the + 3 is for the encoding
        self.block1 = nn.Sequential(
            nn.Linear(l_position * 2 * 3 + 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_size + l_position * 2 * 3 + 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size + 1),
        )  # + sigma

        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_size + l_direction * 6 + 3, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 3),
            nn.Sigmoid(),
        )

        if not with_pos_enc:
            self.embedding_transformation = nn.Linear(3, l_position * 2 * 3 + 3)
            self.embedding_direction = nn.Linear(3, l_direction * 2 * 3 + 3)


        self.l_position = l_position
        self.l_direction = l_direction
        self.hidden_size = hidden_size

    def positional_encoding(self, x, L):
        output = [x]
        for j in range(L):
            output.append(torch.sin(2**j * x))
            output.append(torch.cos(2**j * x))

        return torch.cat(output, dim=1)

    def forward(self, xyz: torch.Tensor, d: torch.Tensor):
        
        # OLder models dont have this attribute
        hack_with_pos_enc = hasattr(self, "with_pos_enc")

        if not hack_with_pos_enc:
            x_embedding = self.positional_encoding(
                xyz, self.l_position
            )  # [batch_size, l_position * 2 * 3 + 3]
            d_embedding = self.positional_encoding(
                d, self.l_direction
            )  # [batch_size, l_direction * 2 * 3 + 3]
        else:
            if self.with_pos_enc:
                x_embedding = self.positional_encoding(
                    xyz, self.l_position
                )  # [batch_size, l_position * 2 * 3 + 3]
                d_embedding = self.positional_encoding(
                    d, self.l_direction
                )  # [batch_size, l_direction * 2 * 3 + 3]
            else:
                x_embedding = self.embedding_transformation(xyz)
                d_embedding = self.embedding_direction(d)

        h = self.block1(x_embedding)
        h = self.block2(torch.cat([h, x_embedding], dim=1))
        sigma = h[:, -1]  # The density
        h = h[:, :-1]
        colors = self.rgb_head(torch.cat([h, d_embedding], dim=1))

        return colors, torch.relu(sigma)

    def intersect(self, x, d):
        return self.forward(x, d)


# Implementing the training loop now

def main():
    
    parser = argparse.ArgumentParser(
        description="Train the model to reconstruct the 3D scene using Nerf"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to the model",
        default=Path("model_nerf.pt"),
    )
    parser.add_argument(
        "--database_path", type=Path, help="Path to the database", default=Path("./fox")
    )
    parser.add_argument(
        "-e",
        "--n-epochs",
        type=int,
        help="Number of epochs to train the model",
        default=15,
    )
    parser.add_argument(
        "-w", "--warmup", type=int, help="Number of warmup epochs", default=1
    )
    parser.add_argument("--no-pos-enc", action="store_true", help="No positional encoding")
    
    args = parser.parse_args()

    model_path = args.model_path
    database_path = args.database_path
    nb_epochs = args.n_epochs
    warmup_epochs = args.warmup
    with_pos_enc = not args.no_pos_enc
    render = False  # render the images PER epoch

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    tn = 8
    tf = 12
    nb_bins = 100

    #device = "cuda"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1024
    num_workers = 30

    rays_o, rays_d, target_px_values = get_rays(Path(database_path))
    # original_shape = rays_o.shape

    # Standard data loader
    training_dataset = torch.cat(
        (rays_o.view(-1, 3), rays_d.view(-1, 3), target_px_values.view(-1, 3)), dim=1
    )
    training_dataloader = DataLoader(
        dataset=training_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Warm up data loader taking only the central part of the data
    # xx = rays_o.reshape(90, 400, 400, 3)

    warmup_dataset = torch.cat(
        (
            rays_o.reshape(90, 400, 400, 3)[:, 100:300, 100:300, :].reshape(-1, 3),
            rays_d.reshape(90, 400, 400, 3)[:, 100:300, 100:300, :].reshape(-1, 3),
            target_px_values.reshape(90, 400, 400, 3)[:, 100:300, 100:300, :].reshape(
                -1, 3
            ),
        ),
        dim=1,
    )
    warmup_dataloader = DataLoader(
        dataset=warmup_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    model = NeRF(with_pos_enc=with_pos_enc).to(device)

    # model = nn.DataParallel(base_model, device_ids=[0, 1, 2, 3])
    # model.intersect = base_model.intersect

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[5, 10], gamma=0.5
    )

    warmup_loss, rendered_images_warm_up = training_loop(
        model,
        warmup_dataloader,
        optimizer,
        scheduler,
        tn,
        tf,
        nb_bins,
        warmup_epochs,
        device=device,
        render=render,
    )

    training_loss, rendered_images_overall = training_loop(
        model,
        training_dataloader,
        optimizer,
        scheduler,
        tn,
        tf,
        nb_bins,
        nb_epochs,
        device=device,
        render=render,
    )

    # Saving the model
    torch.save(model, model_path)

    ## Plotting the warmup and training loss, each one in our subplot
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(warmup_loss)
    ax[0].set_title("Warmup loss")
    ax[1].plot(training_loss)
    ax[1].set_title("Training loss")
    plt.show()
    plt.savefig("loss_nerf.png")

    def plot():
        # Crafting a simple subplot 2x2 to show the images

        # PLEASE, ADAPT THIS TO YOUR NEEDS
        # adust the number of epochs for plotting
        rendered_images = rendered_images_warm_up + rendered_images_overall        
        fig, ax = plt.subplots(2, 2, figsize=(20, 20))
        for i in range(2):
            for j in range(2):
                ax[i, j].axis("off")
                # Setting the title

                ax[i, j].set_title(f"Epoch {i * 2 + j}")
                ax[i, j].imshow(
                    rendered_images[i * 2 + j]
                    .reshape(400, 400, 3)
                    .detach()
                    .cpu()
                    .numpy()
                )
        plt.show()

    # Plotting function for the epochs
    # You can disabled this if you want to see the images
    #plot()


if __name__ == "__main__":
    main()
