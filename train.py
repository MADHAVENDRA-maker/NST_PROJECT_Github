import argparse
import torch
from pathlib import Path
import os
from utils.utils import Imagefolderdataset
from torch.utils.data import DataLoader
from utils.utils import get_transform
from utils.models import VGGEncoder
import torch.optim as optim
from utils.models import Decoder
from tqdm import tqdm
from utils.utils import adaptive_instance_normalization
from utils.utils import calculate_mean_std
from torchvision.utils import save_image




def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_dir",type=str,default="train_dataset_testing",help="Location of content dataset")
    parser.add_argument("--style_dir",type=str,default="test_dataset_testing",help="Location of style dataset")
    parser.add_argument("--vgg",type=str,default="vgg_normalised.pth",help="Location of pre-trained VGG")
    parser.add_argument("--experiment",type=str,default="experiment",help="Name of experiment")
    parser.add_argument("--final_size",type=int,default=256,help="Size of final image")
    parser.add_argument("--content_size",type=int,default=256,help="Size of content image")
    parser.add_argument("--style_size",type=int,default=256,help="Size of style image")
    parser.add_argument("--crop",action="store_true",default=True,help="Crop image")
    parser.add_argument("--batch_size",type=int,default=4,help="Batch size")
    parser.add_argument("--lr",type=float,default=1e-4,help="Learning rate")
    parser.add_argument("--lr_decay",type=float,default=5e-5,help="Learning rate decay")
    parser.add_argument("--epochs",type=int,default=5,help="Number of epochs")
    parser.add_argument("--content_weight",type=float,default=1.0,help="Content weight")
    parser.add_argument("--style_weight",type=float,default=5.0,help="Style weight")
    parser.add_argument("--log_interval",type=int,default=1,help="Log interval")
    parser.add_argument("--save_interval",type=int,default=2,help="Save interval")
    parser.add_argument("--resume",action="store_true",default=False,help="Resume training")
    parser.add_argument("--checkpoint_path",type=str,default=None,help="Path to checkpoint file")
    return parser.parse_args()

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    save_dir = Path("/kaggle/working")/args.experiment
    save_dir.mkdir(exist_ok=True,parents=True)
    with open(save_dir/"arg.txt","w") as arg_file:
        for key,value in vars(args).items():
            arg_file.write(f"{key}: {value}\n")
    content_transform = get_transform(args.content_size,args.crop,args.final_size)
    style_transform = get_transform(args.style_size,args.crop,args.final_size)
    content_dataset = Imagefolderdataset(args.content_dir,content_transform)
    style_dataset = Imagefolderdataset(args.style_dir,style_transform)
    content_dataloader = DataLoader(content_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=torch.cuda.is_available(),drop_last=True,num_workers=2,persistent_workers=True)
    style_dataloader = DataLoader(style_dataset,batch_size=args.batch_size,shuffle=True,pin_memory=torch.cuda.is_available(),drop_last=True,num_workers=2,persistent_workers=True)
    encoder = VGGEncoder(args.vgg).to(device)
    decoder = Decoder().to(device)
    optimizer = optim.Adam(decoder.parameters(),lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: ((1.0)/(1.0 + args.lr_decay*epoch))
    )
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.checkpoint_path,map_location=device)
        decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")
    mse_loss = torch.nn.MSELoss()
    encoder.eval()
    for epoch in range(start_epoch,args.epochs):
        progress_bar = tqdm(zip(content_dataloader,style_dataloader),total=min(len(content_dataloader),len(style_dataloader)))
        running_loss = 0
        running_closs = 0
        running_sloss = 0
        for content_batch,style_batch in progress_bar:
            content_batch = content_batch.to(device)
            style_batch = style_batch.to(device)
            c_feats = encoder(content_batch)
            s_feats = encoder(style_batch)
            t = adaptive_instance_normalization(c_feats[-1],s_feats[-1])
            g = decoder(t)
            g_feats = encoder(g)
            loss_c = mse_loss(g_feats[-1],t)*args.content_weight
            loss_s = 0
            for g_f,s_f in zip(g_feats,s_feats):
                g_mean,g_std = calculate_mean_std(g_f)
                s_mean,s_std = calculate_mean_std(s_f)
                loss_s = loss_s + mse_loss(g_mean,s_mean) + mse_loss(g_std,s_std)
            loss_s = loss_s*args.style_weight
            loss = loss_c + loss_s
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Loss: {loss.item():4f}, Content loss: {loss_c.item():4f}, Style loss: {loss_s.item():4f}")
            running_loss = running_loss + loss.item()
            running_closs = running_closs + loss_c.item()
            running_sloss = running_sloss + loss_s.item()
        scheduler.step()
        num_batches = min(len(content_dataloader),len(style_dataloader))
        running_loss = running_loss/num_batches
        running_closs = running_closs/num_batches
        running_sloss = running_sloss/num_batches
        if (epoch+1)%args.log_interval==0 or (epoch+1)==args.epochs:
            tqdm.write(f"Iter{epoch+1}: Loss: {running_loss:4f} Content loss: {running_closs:4f} Style loss: {running_sloss:4f}")
        if (epoch+1)%args.save_interval==0 or (epoch+1)==args.epochs:
            checkpoint = {"epoch":epoch,"decoder":decoder.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict()}
            torch.save(checkpoint,save_dir/f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint,save_dir/"latest_checkpoint.pth")
            with torch.no_grad():
                output = torch.cat([content_batch,style_batch,g],dim=0)
                save_image(output,save_dir/f"output_{epoch+1}.png",nrow=args.batch_size)

if __name__=="__main__":
    main()