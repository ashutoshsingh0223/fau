# import torch.nn.Module
import torch

class resnet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv = torch.nn.Conv2d(3, 64, 7, 2)
        self.BatchNorm = torch.nn.BatchNorm2d(64)
        self.ReLU = torch.nn.ReLU()
        self.MaxPool = torch.nn.MaxPool2d(3, 2)
        # self.ResBlock1 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, 1),
        #                                      torch.nn.BatchNorm2d(64),
        #                                      torch.nn.ReLU())
        #
        # self.ResBlock2 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, 3, 2),
        #                                      torch.nn.BatchNorm2d(128),
        #                                      torch.nn.ReLU())
        #
        # self.ResBlock3 = torch.nn.Sequential(torch.nn.Conv2d(128, 256, 3, 2),
        #                                      torch.nn.BatchNorm2d(256),
        #                                      torch.nn.ReLU())
        #
        # self.ResBlock4 = torch.nn.Sequential(torch.nn.Conv2d(256, 512, 3, 2),
        #                                      torch.nn.BatchNorm2d(512),
        #                                      torch.nn.ReLU())

        self.ResBlock1 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 3, 1, padding=2),
                                             torch.nn.BatchNorm2d(64),
                                             torch.nn.ReLU(),
                                             torch.nn.Conv2d(64, 64, 3),
                                             torch.nn.BatchNorm2d(64),
                                             torch.nn.ReLU())
        self.res_conv1 = torch.nn.Sequential(torch.nn.Conv2d(64, 64, 1),
                                             torch.nn.BatchNorm2d(64))

        self.ResBlock2 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, 3, 2, padding=3),
                                             torch.nn.BatchNorm2d(128),
                                             torch.nn.ReLU(),
                                             torch.nn.Conv2d(128, 128, 3),
                                             torch.nn.BatchNorm2d(128),
                                             torch.nn.ReLU())
        self.res_conv2 = torch.nn.Sequential(torch.nn.Conv2d(64, 128, 1, 2),
                                             torch.nn.BatchNorm2d(128))

        self.ResBlock3 = torch.nn.Sequential(torch.nn.Conv2d(128, 256, 3, 2, padding=3),
                                             torch.nn.BatchNorm2d(256),
                                             torch.nn.ReLU(),
                                             torch.nn.Conv2d(256, 256, 3),
                                             torch.nn.BatchNorm2d(256),
                                             torch.nn.ReLU())
        self.res_conv3 = torch.nn.Sequential(torch.nn.Conv2d(128, 256, 1, 2),
                                             torch.nn.BatchNorm2d(256))

        self.ResBlock4 = torch.nn.Sequential(torch.nn.Conv2d(256, 512, 3, 2, padding=3),
                                             torch.nn.BatchNorm2d(512),
                                             torch.nn.ReLU(),
                                             torch.nn.Conv2d(512, 512, 3),
                                             torch.nn.BatchNorm2d(512),
                                             torch.nn.ReLU())
        self.res_conv4 = torch.nn.Sequential(torch.nn.Conv2d(256, 512, 1, 2),
                                             torch.nn.BatchNorm2d(512))

        self.GlobalAvgPool = torch.nn.AdaptiveAvgPool2d((1, 1))  # ?
        self.Flatten = torch.nn.Flatten()
        self.FC = torch.nn.Linear(512, 2)

    def forward(self, input_tensor):
        input_tensor = self.Conv(input_tensor)
        input_tensor = self.BatchNorm(input_tensor)
        input_tensor = self.ReLU(input_tensor)
        input_tensor = self.MaxPool(input_tensor)

        input_tensor = self.ResBlock1(input_tensor) + self.res_conv1(input_tensor)
        input_tensor = self.ResBlock2(input_tensor) + self.res_conv2(input_tensor)
        input_tensor = self.ResBlock3(input_tensor) + self.res_conv3(input_tensor)
        input_tensor = self.ResBlock4(input_tensor) + self.res_conv4(input_tensor)

        input_tensor = self.GlobalAvgPool(input_tensor)
        input_tensor = self.Flatten(input_tensor)
        input_tensor = self.FC(input_tensor)

        return input_tensor
