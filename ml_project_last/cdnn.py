import torch

class CDNN(torch.nn.Module):
    def __init__(self,k_size,w,h,channel):
        super(CDNN, self).__init__()
        
        self.activation = torch.nn.ReLU()
        self.dnn1 =  torch.nn.Sequential(
            torch.nn.Conv2d(channel, 64, k_size, stride=1, padding=1),  #  64, 4, 2
            self.activation,
            torch.nn.Conv2d(64, 128, k_size,stride=1, padding=1),       #  64, 4, 2
            self.activation,
            torch.nn.Conv2d(128, 128, k_size,stride=1, padding=1),      #  256, 4, 2
            self.activation,
            torch.nn.Flatten(),
            torch.nn.Linear(128*w*h,1024),
            self.activation,
            torch.nn.Linear(1024,256),
            self.activation,
            torch.nn.Linear(256,1),
        )
    
        
    def forward(self, x):
        out = self.dnn1(x) 
        return out