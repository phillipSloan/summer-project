import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

def double_conv(in_chan, out_chan):
    # padding to keep the image size the same throughout
    conv = nn.Sequential(
        nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )
    return conv


def resize_tensor(tensor, resize_to_this):
    target_height = resize_to_this.size()[2]
    target_width = resize_to_this.size()[3]
    transform = transforms.Resize((target_height, target_width))
    return transform(tensor)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)
    
        self.up_trans_1 = nn.ConvTranspose2d(in_channels=1024,
                                            out_channels=512,
                                            kernel_size=2,
                                            stride=2)
    
        self.up_conv_1 = double_conv(1024, 512)
        
        self.up_trans_2 = nn.ConvTranspose2d(in_channels=512,
                                            out_channels=256,
                                            kernel_size=2,
                                            stride=2)
    
        self.up_conv_2 = double_conv(512, 256)
        
        self.up_trans_3 = nn.ConvTranspose2d(in_channels=256,
                                            out_channels=128,
                                            kernel_size=2,
                                            stride=2)
    
        self.up_conv_3 = double_conv(256, 128)
        
        self.up_trans_4 = nn.ConvTranspose2d(in_channels=128,
                                            out_channels=64,
                                            kernel_size=2,
                                            stride=2)
    
        self.up_conv_4 = double_conv(128, 64)
        
        self.output = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=1,
            padding=1
        )
    
    def forward(self, image):
        # Going down! 
        print(image.shape)
        x = self.down_conv_1(image)
        print(x.shape)
        x1 = x # Storing for skip connection 1
        x = self.max_pooling(x)
        print(x.shape)
        x = self.down_conv_2(x)
        print(x.shape)
        x2 = x # Storing for skip connection 2
        x = self.max_pooling(x)
        print(x.shape)
        x = self.down_conv_3(x)
        print(x.shape)
        x3 = x # Storing for skip connection 3
        x = self.max_pooling(x)
        print(x.shape)
        x = self.down_conv_4(x)
        print(x.shape)
        x4 = x # Storing for skip connection 4
        x = self.max_pooling(x)
        print(x.shape)
        x = self.down_conv_5(x)
        print(x.shape)
        
  
        # and back up again!
        x = self.up_trans_1(x)
        print(x.shape)
        x4 = resize_tensor(x4, x) # Resize x4 to be same size as x
        x = self.up_conv_1(torch.cat([x, x4], 1))
        print(x.shape)
        
        x = self.up_trans_2(x)
        print(x.shape)
        x3 = resize_tensor(x3, x) # Resize x3 to be same size as x
        x = self.up_conv_2(torch.cat([x, x3], 1))
        print(x.shape)
        
        x = self.up_trans_3(x)
        print(x.shape)
        x2 = resize_tensor(x2, x) # Resize x2 to be same size as x
        x = self.up_conv_3(torch.cat([x, x2], 1))
        print(x.shape)
        
        x = self.up_trans_4(x)
        print(x.shape)
        x1 = resize_tensor(x1, x) # Resize x1 to be same size as x
        x = self.up_conv_4(torch.cat([x, x1], 1))
        print(x.shape)
        
        x = self.output(x)
        print(x.shape)
        x = resize_tensor(x,image)
        print(x.shape)
        return x

if __name__ == "__main__":
    image = torch.rand((1,1, 256, 256))
    model = UNet()
    model(image)