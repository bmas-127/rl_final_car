import numpy as np
import torch
import torch.nn as nn

class TActorNetSimple(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, feature_dim: int, N_frame: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(N_frame, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.linear = nn.Sequential(
            nn.Linear(16*(state_dim//8)**2, 256), # = nn.Linear(16*144, 256)
            nn.LayerNorm(256),
            nn.ELU()
        )
        
        self.addition_feature_linear = nn.Sequential(
            nn.Linear(feature_dim * N_frame, 64),
            nn.LayerNorm(64),
            nn.ELU()
        )
        
        self.concat_linear = nn.Sequential(
            nn.Linear(320, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, action_dim)
        )
        

    def forward(self, state, velocity, brake_rate=0.015):
        # state = state.float() / 255.0     
        state_h = self.conv(state)
        state_h = torch.flatten(state_h, start_dim=1)
        state_h = self.linear(state_h) 
        
        v_h = torch.flatten(velocity, start_dim=1)
        v_h = self.addition_feature_linear(v_h)
        
        h = self.concat_linear(torch.concat([state_h, v_h], dim=1))

        h_clone = h.clone()
        # map to valid action space: {steer:[-1, 1], gas:[0, 1], brake:[0, 1]}
        h_clone[:, 0] = (h_clone[:, 0])
        h_clone[:, 1] = (h_clone[:, 1]) 
        
        return h_clone
    
# class CriticNetSimple(nn.Module):
#     def __init__(self, state_dim: int, action_dim: int, feature_dim: int, N_frame: int) -> None:
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(N_frame, 16, kernel_size=3, padding=1),
#             nn.ELU(),
#             nn.Conv2d(16, 16, kernel_size=3, padding=1),
#             nn.ELU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(16, 16, kernel_size=3, padding=1),
#             nn.ELU(),
#             nn.Conv2d(16, 16, kernel_size=3, padding=1),
#             nn.ELU(),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.ELU(),
#             nn.Conv2d(32, 16, kernel_size=3, padding=1),
#             nn.ELU(),
#             nn.MaxPool2d(kernel_size=2),
#         )

#         self.action_linear = nn.Sequential(
#             nn.Linear(action_dim, 256),
#             nn.LayerNorm(256),
#             nn.ELU()
#         )

#         self.state_linear = nn.Sequential(
#             nn.Linear(16*(state_dim//8)**2, 256),
#             nn.LayerNorm(256),
#             nn.ELU(),
#         )
        
#         self.addition_feature_linear = nn.Sequential(
#             nn.Linear(N_frame * feature_dim, 256),
#             nn.LayerNorm(256),
#             nn.ELU(),
#         )

#         self.concat_linear = nn.Sequential(
#             nn.Linear(768, 256),
#             nn.LayerNorm(256),
#             nn.ELU(),
#             nn.Linear(256, 64),
#             nn.LayerNorm(64),
#             nn.ELU(),
#             nn.Linear(64, 1)
#         )

#     def forward(self, state, velocity, action):
#         # extract the state features
#         # state = state.float() / 255.0
#         state_h = self.conv(state)
#         state_h = torch.flatten(state_h, start_dim=1)

#         state_h = self.state_linear(state_h)
#         velocity_h = self.addition_feature_linear(torch.flatten(velocity, start_dim = 1))
#         action_h = self.action_linear(action)

#         # concat
#         h = self.concat_linear(torch.concat((state_h, velocity_h, action_h), dim=1))

#         return h
