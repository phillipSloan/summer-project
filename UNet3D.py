import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

def double_conv(in_chan, out_chan):
    # padding to keep the image size the same throughout
    conv = nn.Sequential(
        nn.Conv3d(in_chan, out_chan, kernel_size=(3,3,3),stride=(1,1,1), padding=1),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_chan, out_chan, kernel_size=(3,3,3),stride=(1,1,1), padding=1),
        nn.ReLU(inplace=True)
    )
    return conv

# for 3D
def resize_tensor(tensor, resize_to_this):
    from torchvision import transforms as transforms
    
    target_size = resize_to_this.size()
    b,c,x,y,z = target_size
    
    new_tensor = torch.empty((b,c,x,y,z))
    # creates a function that resizes a tensor in the HxW (2D) domain 
    # first deals with WxD
    # second transforms HxD - D already done so doesn't matter    
    transform1 = transforms.Resize((y,z))
    transform2 = transforms.Resize((x,z))
    
    for i, t in enumerate(tensor):
        for j, n in enumerate(t):        
            n = transform1(n)
            n = torch.transpose(n, 0, 1)
            n = transform2(n)
            n = torch.transpose(n, 0, 1)
            new_tensor[i][j] = n
    return new_tensor 


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))
        self.down_conv_1 = double_conv(1, 16)
        self.down_conv_2 = double_conv(16, 32)
        self.down_conv_3 = double_conv(32, 64)
        self.down_conv_4 = double_conv(64, 128)
        self.down_conv_5 = double_conv(128, 256)
    
        self.up_trans_1 = nn.ConvTranspose3d(in_channels=256,
                                            out_channels=128,
                                            kernel_size=2,
                                            stride=2)
    
        self.up_conv_1 = double_conv(256, 128)
        
        self.up_trans_2 = nn.ConvTranspose3d(in_channels=128,
                                            out_channels=64,
                                            kernel_size=2,
                                            stride=2)
    
        self.up_conv_2 = double_conv(128, 64)
        
        self.up_trans_3 = nn.ConvTranspose3d(in_channels=64,
                                            out_channels=32,
                                            kernel_size=2,
                                            stride=2)
    
        self.up_conv_3 = double_conv(64, 32)
        
        self.up_trans_4 = nn.ConvTranspose3d(in_channels=32,
                                            out_channels=16,
                                            kernel_size=2,
                                            stride=2)
    
        self.up_conv_4 = double_conv(32, 16)
        
        self.output = nn.Conv3d(
            in_channels=16,
            out_channels=1,
            kernel_size=(1,1,1),
            padding=(1,1,1),
        )
    
    def forward(self, image):
        # Going down! 
        x = self.down_conv_1(image)
        x1 = x # Storing for skip connection 1
        x = self.max_pooling(x)
        x = self.down_conv_2(x)
        x2 = x # Storing for skip connection 2
        x = self.max_pooling(x)
        x = self.down_conv_3(x)
        x3 = x # Storing for skip connection 3
        x = self.max_pooling(x)
        x = self.down_conv_4(x)
        x4 = x # Storing for skip connection 4
        x = self.max_pooling(x)
        x = self.down_conv_5(x)
  
        # and back up again!
        x = self.up_trans_1(x)
        x4 = resize_tensor(x4, x) # Resize x4 to be same size as x
        x = self.up_conv_1(torch.cat([x, x4], 1))
        
        x = self.up_trans_2(x)
        x3 = resize_tensor(x3, x) # Resize x3 to be same size as x
        x = self.up_conv_2(torch.cat([x, x3], 1))
        
        x = self.up_trans_3(x)
        x2 = resize_tensor(x2, x) # Resize x2 to be same size as x
        x = self.up_conv_3(torch.cat([x, x2], 1))
        
        x = self.up_trans_4(x)
        x1 = resize_tensor(x1, x) # Resize x1 to be same size as x
        x = self.up_conv_4(torch.cat([x, x1], 1))

        x = self.output(x)

        x = resize_tensor(x,image)

        return x
    
    
    
if __name__ == "__main__":
    image = torch.rand((1, 1, 197, 233, 189))
    # image = torch.rand((1, 1, 96, 96, 96))
    model = UNet()
    model(image)