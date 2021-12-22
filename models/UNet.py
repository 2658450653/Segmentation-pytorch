import torch.nn as nn
import torch
from torchvision.transforms import transforms
import cv2 as cv
import PIL.Image as Image


class DoubleConv2d(nn.Module):
    def __init__(self, in_c, out_c):
        super(DoubleConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size=3),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class DownLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super(DownLayer, self).__init__()
        self.doubleCon2d = DoubleConv2d(in_c, out_c)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        con = self.doubleCon2d(x)
        return self.maxpool(con), con

class UpLayer(nn.Module):
    def __init__(self, in_c, out_c):
        super(UpLayer, self).__init__()
        self.doubleCon2d = DoubleConv2d(in_c, out_c)
        self.decon = nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=2, stride=2)

    def forward(self, x_up, x_left):
        #print("x", x_up.shape)
        up = self.decon(x_up)
        #print("up", up.shape)
        #print("y", x_left.shape)
        x = self.concate(x_left, up)
        #print("x_cat", x.shape)
        return self.doubleCon2d(x)

    def concate(self, x, y):
        _, c1, w1, h1 = x.shape
        _, c2, w2, h2 = y.shape
        out = torch.cat([x[:, :, (w1 - w2) // 2: w1 - (w1 - w2) // 2, (w1 - w2) // 2: w1 - (w1 - w2) // 2], y], dim=1)
        return out

class UNet(nn.Module):
    def __init__(self, in_c, out_c, out_img_size=224):
        super(UNet, self).__init__()
        self.down1 = DownLayer(in_c, 64)
        self.down2 = DownLayer(64, 128)
        self.down3 = DownLayer(128, 256)
        self.down4 = DownLayer(256, 512)
        self.down5 = DownLayer(512, 1024)

        # 逆卷积，也可以使用上采样(保证k=stride,stride即上采样倍数)
        self.up1 = UpLayer(1024, 512)
        self.up2 = UpLayer(512, 256)
        self.up3 = UpLayer(256, 128)
        self.up4 = UpLayer(128, 64)

        self.conv = nn.Conv2d(64, out_c, 1)

        self.resize_in = transforms.Resize([572, 572])
        self.resize_out = transforms.Resize([out_img_size, out_img_size])

    def forward(self, x):

        x = self.resize_in(x)

        d1, r1 = self.down1(x)
        #print("MP and 2Con out:", d1.shape, r1.shape)
        d2, r2 = self.down2(d1)
        #print("MP and 2Con out:", d2.shape, r2.shape)
        d3, r3 = self.down3(d2)
        #print("MP and 2Con out:", d3.shape, r3.shape)
        d4, r4 = self.down4(d3)
        #print("MP and 2Con out:", d4.shape, r4.shape)
        d5, r5 = self.down5(d4)
        #print("MP and 2Con out:", d5.shape, r5.shape)

        u1 = self.up1(r5, r4)
        #print("up out:", u1.shape)
        u2 = self.up2(u1, r3)
        #print("up out:", u2.shape)
        u3 = self.up3(u2, r2)
        #print("up out:", u3.shape)
        u4 = self.up4(u3, r1)
        #print("up out:", u4.shape)

        out =  self.conv(u4)

        out = self.resize_out(out)

        return out


if __name__ == "__main__":
    x = torch.rand(1, 3, 572, 572, dtype=torch.float32)
    model = UNet(3, 3)
    out = model(x)
    out = out.cpu().detach()
    resize = transforms.Resize([224, 224])
    #print(out.shape)

    x = cv.imread("test.jpg")
    x = Image.fromarray(x)
    test_transforms = transforms.Compose(
        [
            transforms.Resize(500),
            transforms.CenterCrop(448),
            transforms.ToTensor(),
        ]
    )
    x = test_transforms(x)
    x = x.unsqueeze(0)
    out = model(x)
    out = out.cpu().detach().numpy().squeeze(0).transpose(-2, -1, -3)
    print(out.shape)
    cv.imshow("test", out)
    cv.waitKey(0)
