import pandas as pd
import os
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from deepspeed.accelerator import get_accelerator
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
import deepspeed
import argparse

path = 'face2comics_v1.0.0_by_Sxela/face2comics_v1.0.0_by_Sxela/'
comics_path = path + 'comics/'
face_path = path + 'face/'

comics = []
face = []
for i in os.listdir(comics_path):
    comics.append(comics_path + i)

for i in os.listdir(face_path):
    face.append(face_path + i)


class UNet(nn.Module):
    def __init__(self, args):
        super(UNet, self).__init__()
        # Downsampling
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


face_train = face[:8000]
comics_train = comics[:8000]
face_test = face[8000:]
comics_test = comics[8000:]
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

class PairedImageDataset(Dataset):
    def __init__(self, face_paths, comic_paths, transform=None):
        self.face_paths = face_paths
        self.comic_paths = comic_paths
        self.transform = transform

    def __len__(self):
        return len(self.face_paths)

    def __getitem__(self, idx):
        face_image = Image.open(self.face_paths[idx]).convert("RGB")
        comic_image = Image.open(self.comic_paths[idx]).convert("RGB")
        
        if self.transform:
            face_image = self.transform(face_image)
            comic_image = self.transform(comic_image)
        
        return face_image, comic_image

def load_train_objs():
    train_set = PairedImageDataset(face_paths=face_train, comic_paths=comics_train, transform=transform)  
    # model = UNet()
    # optimizer = optim.Adam(model.parameters(), lr=0.00005)
    return train_set#, model, optimizer

def load_test_objs():
    test_set = PairedImageDataset(face_paths=face_test, comic_paths=comics_test, transform=transform)
    return test_set
   
def add_argument():
    parser = argparse.ArgumentParser(description="CIFAR")

    # For train.
    parser.add_argument(
        "-e",
        "--epochs",
        default=30,
        type=int,
        help="number of total epochs (default: 30)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=16,
        type=int,
        help=""
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=2000,
        help="output logging information at a given interval",
    )

    # For mixed precision training.
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="Datatype used for training",
    )

    # For ZeRO Optimization.
    parser.add_argument(
        "--stage",
        default=0,
        type=int,
        choices=[0, 1, 2, 3],
        help="Datatype used for training",
    )

    # For MoE (Mixture of Experts).
    parser.add_argument(
        "--moe",
        default=False,
        action="store_true",
        help="use deepspeed mixture of experts (moe)",
    )
    parser.add_argument(
        "--ep-world-size", default=1, type=int, help="(moe) expert parallel world size"
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        nargs="+",
        default=[
            1,
        ],
        help="number of experts list, MoE related.",
    )
    parser.add_argument(
        "--mlp-type",
        type=str,
        default="standard",
        help="Only applicable when num-experts > 1, accepts [standard, residual]",
    )
    parser.add_argument(
        "--top-k", default=1, type=int, help="(moe) gating top 1 and 2 supported"
    )
    parser.add_argument(
        "--min-capacity",
        default=0,
        type=int,
        help="(moe) minimum capacity of an expert regardless of the capacity_factor",
    )
    parser.add_argument(
        "--noisy-gate-policy",
        default=None,
        type=str,
        help="(moe) noisy gating (only supported with top-1). Valid values are None, RSample, and Jitter",
    )
    parser.add_argument(
        "--moe-param-group",
        default=False,
        action="store_true",
        help="(moe) create separate moe param groups, required when using ZeRO w. MoE",
    )

    # Include DeepSpeed configuration arguments.
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()

    return args



def get_ds_config(args):
    """Get the DeepSpeed configuration dictionary."""
    ds_config = {
        "train_batch_size": args.batch_size,
        "steps_per_print": 2000,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.0001,
                "betas": [0.8, 0.999],
                "eps": 1e-8,
                "weight_decay": 3e-7,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 0.001,
                "warmup_num_steps": 1000,
            },
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "bf16": {"enabled": args.dtype == "bf16"},
        # "fp16": {
        #     "enabled": args.dtype == "fp16",
        #     "fp16_master_weights_and_grads": False,
        #     "loss_scale": 0,
        #     "loss_scale_window": 500,
        #     "hysteresis": 2,
        #     "min_loss_scale": 1,
        #     "initial_scale_power": 15,
        # },
        "wall_clock_breakdown": False,
        "zero_optimization": {
            "stage": args.stage,
            "allgather_partitions": True,
            "reduce_scatter": True,
            "allgather_bucket_size": 50000000,
            "reduce_bucket_size": 50000000,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "cpu_offload": False,
        },
    }
    return ds_config




def test(model_engine, testset, local_device, target_dtype, ):
    """Test the network on the test data.

    Args:
        model_engine (deepspeed.runtime.engine.DeepSpeedEngine): the DeepSpeed engine.
        testset (torch.utils.data.Dataset): the test dataset.
        local_device (str): the local device name.
        target_dtype (torch.dtype): the target datatype for the test data.
        test_batch_size (int): the test batch size.

    """

    # Define the test dataloader.
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    total_loss = 0

    # Start testing.
    model_engine.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if target_dtype != None:
                images = images.to(target_dtype)
            outputs = model_engine(images.to(local_device))
            loss = model_engine.criterion(outputs, labels)
            total_loss += loss


    if model_engine.local_rank == 0:
        print(
            f"Loss of the network on the {len(testloader)} test images: {total_loss / len(testloader) : .4f} %"
        )

def main(args):
    # Initialize DeepSpeed distributed backend.
    os.environ["NCCL_DEBUG"]="INFO"
    os.environ["NCCL_SOCKET_IFNAME"]="eno1"
    deepspeed.init_distributed(init_method='tcp://192.168.1.11:12355', rank=0, world_size=1)
    _local_rank = 0 #int(os.environ.get("LOCAL_RANK"))
    get_accelerator().set_device(_local_rank)

    ########################################################################
    # Step1. Data Preparation.
    #
    # The output of torchvision datasets are PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1].
    #
    # Note:
    #     If running on Windows and you get a BrokenPipeError, try setting
    #     the num_worker of torch.utils.data.DataLoader() to 0.
    ########################################################################
    

    if torch.distributed.get_rank() != 0:
        # Might be downloading cifar data, let rank 0 download first.
        print("yo FIRST BARRIER")
        torch.distributed.barrier(device_ids=[0])
        print("yoo FIRST BARRIER")

    # Load or download cifar data.
    trainset = load_train_objs()
    testset = load_test_objs()

    if torch.distributed.get_rank() == 0:
    # Cifar data is downloaded, indicate other ranks can proceed.
        print("yo")
        torch.distributed.barrier()
        print("yoo")

    ########################################################################
    # Step 2. Define the network with DeepSpeed.
    #
    # First, we define a Convolution Neural Network.
    # Then, we define the DeepSpeed configuration dictionary and use it to
    # initialize the DeepSpeed engine.
    ########################################################################
    net = UNet(args)

    # Get list of parameters that require gradients.
    parameters = filter(lambda p: p.requires_grad, net.parameters())

    # Initialize DeepSpeed to use the following features.
    #   1) Distributed model.
    #   2) Distributed data loader.
    #   3) DeepSpeed optimizer.
    ds_config = get_ds_config(args)
    # print("Size of trainset:", len(trainset))
    model_engine, optimizer, trainloader, __ = deepspeed.initialize(
        args=args,
        model=net,
        model_parameters=parameters,
        training_data=trainset,
        config=ds_config,
    )

    # Get the local device name (str) and local rank (int).
    local_device = get_accelerator().device_name(model_engine.local_rank)
    print('Device Used:', local_device)
    local_rank = model_engine.local_rank

    # For float32, target_dtype will be None so no datatype conversion needed.
    target_dtype = None
    if model_engine.bfloat16_enabled():
        target_dtype = torch.bfloat16
    elif model_engine.fp16_enabled():
        target_dtype = torch.half

    # Define the Classification Cross-Entropy loss function.
    criterion = nn.MSELoss()

    ########################################################################
    # Step 3. Train the network.
    #
    # This is when things start to get interesting.
    # We simply have to loop over our data iterator, and feed the inputs to the
    # network and optimize. (DeepSpeed handles the distributed details for us!)
    ########################################################################

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        print(len(trainloader))
        for i, data in tqdm(enumerate(trainloader)):
            # Get the inputs. ``data`` is a list of [inputs, labels].
            inputs, labels = data[0].to(local_device), data[1].to(local_device)

            # Try to convert to target_dtype if needed.
            if target_dtype != None:
                inputs = inputs.to(target_dtype)

            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)
            # print("LOSS:",loss)

            model_engine.backward(loss)
            model_engine.step()

            # Print statistics
            running_loss += loss.item()
            if local_rank == 0 and i % args.log_interval == (
                args.log_interval - 1
            ):  # Print every log_interval mini-batches.
                print(
                    f"[{epoch + 1 : d}, {i + 1 : 5d}] loss: {running_loss / args.log_interval : .3f}"
                )
                # print("Inputs", inputs.shape, '\nOutputs', outputs.shape)
                running_loss = 0.0
    print("Finished Training")

    ########################################################################
    # Step 4. Test the network on the test data.
    ########################################################################
    test(model_engine, testset, local_device, target_dtype)


if __name__ == "__main__":
    args = add_argument()
    main(args)

    






