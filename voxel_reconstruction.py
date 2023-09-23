from pathlib import Path
import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import argparse

from volumetric_rendering import rendering
#from sample_from_model import test


def render_image(index, model, rays_o, rays_d, tn, tf, n_bins=100, device="cpu"):
    with torch.no_grad():
        return rendering(
            model,
            rays_o[index].to(device),
            rays_d[index].to(device),
            tn,
            tf,
            nbins=n_bins,
            device=device,
        )


def get_rays(datapath: Path, mode: str = "train"):
    pose_file_names = sorted(datapath.glob(f"{mode}/pose/*.txt"))
    intrinsics_file_names = sorted(datapath.glob(f"{mode}/intrinsics/*.txt"))
    # image_file_names = sorted(datapath.glob("./imgs/*.png"))

    # CHecking if the file names are in the same order
    #
    for i, (pose_file_name, intrinsic_file_name) in enumerate(
        zip(pose_file_names, intrinsics_file_names)
    ):
        assert (
            pose_file_name.stem == intrinsic_file_name.stem
        ), f"Files {pose_file_name.stem} and {intrinsic_file_name.stem} are not in the same order"

    assert len(pose_file_names) == len(intrinsics_file_names)

    # Read the data
    N = len(pose_file_names)
    poses = np.zeros(
        (N, 4, 4)
    )  # 4,4 matrix, 3x3 for rotation and 1x3 for translation, plus 1 for homogenous coordinates
    intrinsics = np.zeros(
        (N, 4, 4)
    )  # 4,4 matrix, 3x3 for rotation and 1x3 for translation, plus 1 for homogenous coordinates
    images = []

    for i, (pose_file_name, intrinsic_file_name) in enumerate(
        zip(pose_file_names, intrinsics_file_names)
    ):

        image_file_name = datapath / f"imgs/{pose_file_name.stem}.png"

        assert image_file_name.exists(), f"File {image_file_name} does not exist"

        # Read pose
        pose = open(pose_file_name).read().split()
        pose = np.array(pose, dtype="float").reshape(4, 4)
        poses[i] = pose

        # Read intrinsic
        intrinsic = open(intrinsic_file_name).read().split()
        intrinsic = np.array(intrinsic, dtype="float").reshape(4, 4)
        intrinsics[i] = intrinsic

        # Read images
        img = imageio.imread(image_file_name) / 255.0
        images.append(img[None, ...])

    images = np.concatenate(images, axis=0)

    # Getting rid of the alpha channel
    if images.shape[3] == 4:
        images = images[..., :3] * images[..., -1:] + 1 - images[..., -1:]

    H, W = images.shape[1], images.shape[2]

    # Creating the rays
    rays_o = np.zeros((N, H * W, 3))
    rays_d = np.zeros((N, H * W, 3))
    target_px_values = images.reshape(N, H * W, 3)

    for i in range(N):
        c2w = poses[i]  # Camera angle to world
        f = intrinsics[
            i, 0, 0
        ]  # Focal length is the first element of the intrinsic matrix

        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)

        dirs = np.stack((u - W / 2, -(v - H / 2), -np.ones_like(u) * f), axis=-1)

        # Applying the camera transformation
        dirs = (c2w[:3, :3] @ dirs[..., None]).squeeze(-1)
        dirs = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

        rays_d[i] = dirs.reshape(-1, 3)
        rays_o[i] += c2w[:3, 3]

    rays_o = torch.from_numpy(rays_o).float()
    rays_d = torch.from_numpy(rays_d).float()
    target_px_values = torch.from_numpy(target_px_values).float()

    return rays_o, rays_d, target_px_values


class Voxels(nn.Module):

    def __init__(self, nb_voxels: int = 100, device: str = "cpu", scale: float = 1.0):
        super(Voxels, self).__init__()
        self.voxels = nn.Parameter(
            torch.rand(nb_voxels, nb_voxels, nb_voxels, 4, device=device),
            requires_grad=True,
        )
        # .to(device)
        self.device = device
        self.nb_voxels = nb_voxels
        self.scale = scale

    def forward(
        self, xyz: torch.Tensor, d
    ):  # d is the direction which will not be used by this class

        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        # Check if the point is inside the voxel grid
        cond = (
            (x.abs() < (self.scale / 2))
            & (y.abs() < (self.scale / 2))
            & (z.abs() < (self.scale / 2))
        )

        # If the point is outside the voxel grid, we return a black color
        i_x = (x[cond] / (self.scale / self.nb_voxels)) + self.nb_voxels // 2
        i_y = (y[cond] / (self.scale / self.nb_voxels)) + self.nb_voxels // 2
        i_z = (z[cond] / (self.scale / self.nb_voxels)) + self.nb_voxels // 2
        i_x = i_x.long()
        i_y = i_y.long()
        i_z = i_z.long()

        # Clamp the values to be inside the voxel grid
        i_x, i_y, i_z = (
            i_x.clamp(0, self.nb_voxels - 1),
            i_y.clamp(0, self.nb_voxels - 1),
            i_z.clamp(0, self.nb_voxels - 1),
        )

        colors_and_density = torch.zeros((xyz.shape[0], 4)).to(self.device)
        colors_and_density[cond, :3] = self.voxels[i_x, i_y, i_z, :3]
        colors_and_density[cond, -1] = self.voxels[i_x, i_y, i_z, -1]

        colors = colors_and_density[:, :3]
        density = colors_and_density[:, -1]

        # Little hack here to make color between 0 and 1 and
        # have a positive density
        colors = torch.sigmoid(colors)
        density = torch.relu(density)

        return colors, density

    def intersect(
        self, x, d
    ):  # d is the direction which will not be used by this class
        return self.forward(x, d)


# Implementing the training loop now


def training_loop(
    model,
    dataloader,
    optimizer,
    scheduler,
    tn,
    tf,
    nb_bins,
    nb_epochs,
    device="cpu",
    render=False,
    database_path="./fox",
):

    training_loss = []
    rendering_per_epoch = []

    # not my best moment here in terms of code quality
    if render:
        rays_o, rays_d, _ = get_rays(Path(database_path))
        rays_o = rays_o.to(device)
        rays_d = rays_d.to(device)

    for epoch in range(nb_epochs):
        for i, batch in tqdm(enumerate(dataloader)):
            o = batch[:, :3].to(device)
            d = batch[:, 3:6].to(device)
            target = batch[:, 6:].to(device)

            # Forward by getting the rendering given by the model
            prediction = rendering(model, o, d, tn, tf, nb_bins, device)

            loss = (prediction - target).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
            
            #if i==100:
                #break

        scheduler.step()

        if render:
            #import ipdb; ipdb.set_trace()
            # Rendering on the CPU.
            # It sucks a lot of memory

            img = render_image(0, model.to("cpu"), rays_o, rays_d, tn, tf, nb_bins, device="cpu")
            # Putting the image back to the original device
            # Yes, I know, not my best moment here
            model = model.to(device)

            #img = test(model, rays_o[0], rays_d[0], tn, tf, nb_bins, device=device)

            #img = render_image(model, rays_o[0], rays_d[0], tn, tf, n_bins=100, chunk_size=10, H=400, W=400)
            rendering_per_epoch.append(img)

        # if i % 100 == 0:
        #    print(f"Epoch {epoch}, iteration {i}, loss {loss.item()}")
    return training_loss, rendering_per_epoch


def main():
    parser = argparse.ArgumentParser(
        description="Train the model to reconstruct the 3D scene using voxels"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Path to the model",
        default=Path("model_voxel.pt"),
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

    args = parser.parse_args()

    model_path = args.model_path
    database_path = args.database_path
    nb_epochs = args.n_epochs
    warmup_epochs = args.warmup

    render = False  # render the images PER epoch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tn = 8
    tf = 12
    nb_bins = 100

    batch_size = 1024
    num_workers = 30

    rays_o, rays_d, target_px_values = get_rays(Path(database_path))
    original_shape = rays_o.shape

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

    model = Voxels(scale=3, device=device)
    # base_model = Voxels(scale=3.0, device=device)
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
    plt.savefig("loss_voxel.png")

    def plot():
        # Crafting a simple subplot 4x4 to show the images
        rendered_images = rendered_images_warm_up + rendered_images_overall
        fig, ax = plt.subplots(4, 4, figsize=(20, 20))
        for i in range(4):
            for j in range(4):
                ax[i, j].axis("off")
                # Setting the title

                ax[i, j].set_title(f"Epoch {i * 4 + j}")
                ax[i, j].imshow(
                    rendered_images[i * 4 + j]
                    .reshape(400, 400, 3)
                    .detach()
                    .cpu()
                    .numpy()
                )
        plt.show()

    # Plotting function for the epochs
    # You can disabled this if you want to see the images
    # plot()


if __name__ == "__main__":
    main()
