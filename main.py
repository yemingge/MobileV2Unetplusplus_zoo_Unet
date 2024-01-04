import torch
import argparse
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from dataset import LiverDataset
from common_tools import transform_invert
from torch.utils.tensorboard import SummaryWriter
from thop import profile
import datetime
from acc import *
from numpy import asarray

from tkinter import ttk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

from resunetpp import resunetpp
from new_unet import NestedUNet, U_Net
from mobileunet import mobileUNet
# from mobilev3pp import mobileunetv3pp
from mobilev2pp import mobileunetv2pp


def makedir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.ToTensor()
y_transforms = transforms.ToTensor()


# x_transforms =transforms.Compose([
#         transforms.RandomResizedCrop(256),
#         transforms.ToTensor(),
#     ])
# y_transforms =transforms.Compose([
#         transforms.RandomResizedCrop(256),
#         transforms.ToTensor(),
#     ])


def train_model(model, criterion, optimizer, dataload, net_class, num_epochs=100):
    makedir('./model/model_' + net_class)
    writer = SummaryWriter("train/" + net_class)
    start_epoch = 0
    lossmin = 0
    for epoch in range(start_epoch + 1, num_epochs):
        print(net_class + '_Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss / step))
        writer.add_scalar("train_loss_" + net_class, epoch_loss / step, epoch)
        if epoch == 1:
            lossmin = epoch_loss / step
        if lossmin >= epoch_loss / step:
            torch.save(model.state_dict(), './model/model_' + net_class + '/weights_%d.pth' % epoch)
            lossmin = epoch_loss / step

        valid_dataset = LiverDataset("data/val", transform=x_transforms, target_transform=y_transforms)
        valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True)
        if (epoch + 2) % 1 == 0:
            loss_val = 0.
            model.eval()
            with torch.no_grad():
                step_val = 0
                for x, y in valid_loader:
                    step_val += 1
                    x = x.type(torch.FloatTensor)
                    inputs = x.to(device)
                    labels = y.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss_val += loss.item()

                print("epoch %d valid_loss:%0.3f" % (epoch, loss_val / step_val))
                writer.add_scalar("val_loss_" + net_class, loss_val / step_val, epoch)


def train(args):
    net_class = args.net
    if net_class == "unet":
        model = U_Net().to(device)
    elif net_class == "unetpp":
        model = NestedUNet().to(device)
    elif net_class == "resunetpp":
        model = resunetpp().to(device)
    elif net_class == "mobileunet":
        model = mobileUNet(1, 1).to(device)
    elif net_class == "mobilev2unetpp":
        model = mobileunetv2pp().to(device)
    # elif net_class == "mobilev3unetpp":
    #     model = mobileunetv3pp().to(device)

    batch_size = args.batch_size
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset("./data/train", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders, net_class)


def test(args):
    net_class = args.net
    if net_class == "unet":
        model = U_Net()
    elif net_class == "unetpp":
        model = NestedUNet()
    elif net_class == "resunetpp":
        model = resunetpp()
    elif net_class == "mobileunet":
        model = mobileUNet(1, 1)
    elif net_class == "mobilev2unetpp":
        model = mobileunetv2pp()
    # elif net_class == "mobilev3unetpp":
    #     model = mobileunetv3pp()

    liver_dataset = LiverDataset("data/val", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)

    flops, params = profile(model, (1, 1, 256, 256))
    print('flops: ', flops, 'params: ', params)

    model.load_state_dict(torch.load(args.ckpt, map_location='cuda'), strict=False)

    makedir('./predict/pre_' + net_class)
    save_root = './predict/pre_' + net_class

    model.eval()
    plt.ion()
    index = 0
    time_temp = datetime.timedelta()
    with torch.no_grad():
        for x, ground in dataloaders:
            x = x.type(torch.FloatTensor)

            starttime = datetime.datetime.now()
            y = model(x)
            endtime = datetime.datetime.now()
            # temp=(endtime - starttime).seconds
            time_temp += (endtime - starttime)

            x = torch.squeeze(x)
            x = x.unsqueeze(0)
            ground = torch.squeeze(ground)
            ground = ground.unsqueeze(0)
            img_ground = transform_invert(ground, y_transforms)
            img_x = transform_invert(x, x_transforms)
            img_y = torch.squeeze(y).numpy()
            src_path = os.path.join(save_root, "predict_%d_s.png" % index)
            save_path = os.path.join(save_root, "predict_%d_o.png" % index)
            ground_path = os.path.join(save_root, "predict_%d_g.png" % index)
            img_ground.save(ground_path)
            # img_x.save(src_path)
            cv2.imwrite(save_path, img_y * 255)
            index = index + 1
    print(time_temp / 31)


def test_app():
    class ImageProcessingApp:
        def __init__(self, master):
            self.master = master
            self.master.title("Unet-Zoo for segmentation")
            self.model = None
            self.file_path = None
            menu_bar = tk.Menu(self.master)
            self.master.config(menu=menu_bar)

            file_menu = tk.Menu(menu_bar, tearoff=0)
            menu_bar.add_cascade(label="文件", menu=file_menu)
            file_menu.add_command(label="打开图片", command=self.open_image)
            file_menu.add_separator()
            file_menu.add_command(label="退出", command=self.master.destroy)

            self.net_class_label = tk.Label(self.master, text="选择网络模型：", font=("Arial", 12))
            self.net_class_label.pack(pady=10)

            self.net_options = ["UNet", "UNet++", "ResUNet++", "MobileUNet", "MobileV2UNet++(ours)"]

            self.net_class_var = tk.StringVar()
            self.net_class_combobox = ttk.Combobox(self.master, textvariable=self.net_class_var,
                                                   values=self.net_options)
            self.net_class_combobox.pack(pady=10)
            self.net_class_combobox.bind("<<ComboboxSelected>>", self.on_combobox_selected)

            self.original_canvas = tk.Canvas(self.master, width=256, height=300)
            self.original_canvas.pack(side=tk.LEFT, padx=10, expand="yes")

            self.processed_canvas = tk.Canvas(self.master, width=256, height=300)
            self.processed_canvas.pack(side=tk.LEFT, padx=10, expand="yes")

            self.true_canvas = tk.Canvas(self.master, width=256, height=300)
            self.true_canvas.pack(side=tk.LEFT, padx=10, expand="yes")

            self.time_label = tk.Label(self.master, text=" ", font=("Arial", 12))
            self.time_label.place(relx=0, anchor="nw")

            self.mIoU_label = tk.Label(self.master, text=" ", font=("Arial", 12))
            self.mIoU_label.place(relx=1, anchor="ne")

        def on_combobox_selected(self, event):
            net_class = self.net_class_var.get()
            print("Selected option:", net_class)
            if net_class == "UNet":
                self.model = U_Net()
                self.model.load_state_dict(torch.load(".\model\model_unet\weights_88.pth", map_location='cuda'),
                                           strict=False)
            elif net_class == "UNet++":
                self.model = NestedUNet()
                self.model.load_state_dict(torch.load(".\model\model_unetpp\weights_99.pth", map_location='cuda'),
                                           strict=False)
            elif net_class == "ResUNet++":
                self.model = resunetpp()
                self.model.load_state_dict(torch.load(".\model\model_resunetpp\weights_24.pth", map_location='cuda'),
                                           strict=False)
            elif net_class == "MobileUNet":
                self.model = mobileUNet(1, 1)
                self.model.load_state_dict(torch.load(".\model\model_mobileunet\weights_96.pth", map_location='cuda'),
                                           strict=False)
            elif net_class == "MobileV2UNet++(ours)":
                self.model = mobileunetv2pp()
                self.model.load_state_dict(
                    torch.load(".\model\model_mobilev2unetpp\weights_64.pth", map_location='cuda'),
                    strict=False)
            self.open_image_2()

        def open_image(self):
            self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
            if self.file_path != None:
                # 加载原始图像
                original_image = Image.open(self.file_path)

                # 在原始图像的画布上显示
                self.display_image(original_image, canvas=self.original_canvas)

                true_image_path = os.path.join("E:\\MobileV2Unet++\\base\\data\\val\\Tru",
                                               os.path.basename(self.file_path))
                true_image = Image.open(true_image_path)

                self.display_image(true_image, canvas=self.true_canvas)

                processed_image = self.process_image(original_image, self.model, true_image)

                self.display_image(processed_image, canvas=self.processed_canvas)

                self.original_canvas.create_text((128, 256), text="原始图片", anchor="n", font=("Arial", 12))
                self.processed_canvas.create_text((128, 256), text="预测分割图片", anchor="n", font=("Arial", 12))
                self.true_canvas.create_text((128, 256), text="真实分割图片", anchor="n", font=("Arial", 12))

        def open_image_2(self):
            if self.file_path != None:
                # 加载原始图像
                original_image = Image.open(self.file_path)

                # 在原始图像的画布上显示
                self.display_image(original_image, canvas=self.original_canvas)

                true_image_path = os.path.join("E:\\MobileV2Unet++\\base\\data\\val\\Tru",
                                               os.path.basename(self.file_path))
                true_image = Image.open(true_image_path)

                self.display_image(true_image, canvas=self.true_canvas)

                processed_image = self.process_image(original_image, self.model, true_image)

                self.display_image(processed_image, canvas=self.processed_canvas)

                self.original_canvas.create_text((128, 256), text="原始图片", anchor="n", font=("Arial", 12))
                self.processed_canvas.create_text((128, 256), text="预测分割图片", anchor="n", font=("Arial", 12))
                self.true_canvas.create_text((128, 256), text="真实分割图片", anchor="n", font=("Arial", 12))

        def process_image(self, image, model, true_image):
            image = image.convert('L')
            transform = transforms.ToTensor()
            input_tensor = transform(image)
            input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.type(torch.FloatTensor)

            model.eval()
            with torch.no_grad():
                starttime = datetime.datetime.now()
                output_tensor = model(input_tensor)
                endtime = datetime.datetime.now()
                time = (endtime - starttime).total_seconds()
                self.time_label["text"] = f"推理用时: {time} 秒"

            output_tensor = torch.squeeze(output_tensor).numpy()

            cv2.imwrite("im_save.png", output_tensor * 255)
            processed_image = Image.open("im_save.png")

            imgLabel = asarray(true_image.convert('L'))
            imgPredict = asarray(processed_image)
            imgPredict = np.where(imgPredict >= 1, 1, imgPredict)
            imgLabel = np.where(imgLabel >= 1, 1, imgLabel)
            metric = SegmentationMetric(2)
            metric.addBatch(imgPredict, imgLabel)

            mIoU = metric.meanIntersectionOverUnion()
            self.mIoU_label["text"] = f"mIoU: {mIoU:.4f} "

            return processed_image

        def display_image(self, image, canvas):
            canvas.delete("all")
            tk_image = ImageTk.PhotoImage(image)
            canvas.create_image(0, 0, anchor="nw", image=tk_image)
            canvas.image = tk_image

    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()


if __name__ == '__main__':
    seed = 2024
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    supported_nets = ["unet", "unetpp", "resunetpp", "mobileunet", "mobilev2unetpp"]
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, help="train, test", default="train")
    parse.add_argument("--batch_size", type=int, default=4)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file",
                       default=".\model\model_mobilev2unetpp\weights_64.pth")
    parse.add_argument("--net", type=str, help="unet,unetpp,resunetpp,mobileunet,mobilev2unetpp",
                       default="mobilev2unetpp")
    args = parse.parse_args()

    # for net_value in supported_nets:
    #     args.net = net_value
    #     train(args)

    if args.action == "train":
        train(args)
    elif args.action == "test":
        test(args)
