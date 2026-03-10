import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utilities3 import *

def set_seed(seed):    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
set_seed(0)

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, s1=32, s2=32, L = (1.0, 1.0)):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.s1 = s1
        self.s2 = s2
        self.L1 = float(L[0])
        self.L2 = float(L[1])

        # Use randn (mean 0) instead of rand
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, u, x_in=None, x_out=None, iphi=None, code=None):
        batchsize = u.shape[0]

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        if x_in == None:
            u_ft = torch.fft.rfft2(u)
            s1 = u.size(-2)
            s2 = u.size(-1)
        else:
            u_ft = self.fft2d(u, x_in, iphi, code)
            s1 = self.s1
            s2 = self.s2

        # Multiply relevant Fourier modes
        # print(u.shape, u_ft.shape)
        factor1 = self.compl_mul2d(u_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        factor2 = self.compl_mul2d(u_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        if x_out == None:
            out_ft = torch.zeros(batchsize, self.out_channels, s1, s2 // 2 + 1, dtype=torch.cfloat, device=u.device)
            out_ft[:, :, :self.modes1, :self.modes2] = factor1
            out_ft[:, :, -self.modes1:, :self.modes2] = factor2
            u = torch.fft.irfft2(out_ft, s=(s1, s2))
        else:
            out_ft = torch.cat([factor1, factor2], dim=-2)
            u = self.ifft2d(out_ft, x_out, iphi, code)

        return u

    def fft2d(self, u, x_in, iphi=None, code=None):
        # u (batch, channels, n)
        # x_in (batch, n, 2) locations in [0,1]*[0,1]
        # iphi: function: x_in -> x_c

        batchsize = x_in.shape[0]
        N = x_in.shape[1]
        device = x_in.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                            torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        # print(x_in.shape)
        if iphi == None:
            x = x_in
        else:
            x = iphi(x_in, code)

        # print(x.shape)
        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[...,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[...,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1/self.L1 + K2/self.L2

        # basis (batch, N, m1, m2)
        basis = torch.exp(-1j * 2 * np.pi * K).to(device)

        # Y (batch, channels, N)
        u = u + 0j
        Y = torch.einsum("bcn,bnxy->bcxy", u, basis)
        return Y

    def ifft2d(self, u_ft, x_out, iphi=None, code=None):
        # u_ft (batch, channels, kmax, kmax)
        # x_out (batch, N, 2) locations in [0,1]*[0,1]
        # iphi: function: x_out -> x_c

        batchsize = x_out.shape[0]
        N = x_out.shape[1]
        device = x_out.device
        m1 = 2 * self.modes1
        m2 = 2 * self.modes2 - 1

        # wavenumber (m1, m2)
        k_x1 =  torch.cat((torch.arange(start=0, end=self.modes1, step=1), \
                            torch.arange(start=-(self.modes1), end=0, step=1)), 0).reshape(m1,1).repeat(1,m2).to(device)
        k_x2 =  torch.cat((torch.arange(start=0, end=self.modes2, step=1), \
                            torch.arange(start=-(self.modes2-1), end=0, step=1)), 0).reshape(1,m2).repeat(m1,1).to(device)

        if iphi == None:
            x = x_out
        else:
            x = iphi(x_out, code)

        # K = <y, k_x>,  (batch, N, m1, m2)
        K1 = torch.outer(x[:,:,0].view(-1), k_x1.view(-1)).reshape(batchsize, N, m1, m2)
        K2 = torch.outer(x[:,:,1].view(-1), k_x2.view(-1)).reshape(batchsize, N, m1, m2)
        K = K1/self.L1 + K2/self.L2

        # basis (batch, N, m1, m2)
        basis = torch.exp(1j * 2 * np.pi * K).to(device)

        # coeff (batch, channels, m1, m2)
        u_ft2 = u_ft[..., 1:].flip(-1, -2).conj()
        u_ft = torch.cat([u_ft, u_ft2], dim=-1)

        # Y (batch, channels, N)
        Y = torch.einsum("bcxy,bnxy->bcn", u_ft, basis)
        Y = Y.real
        return Y


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels, out_channels, is_mesh=True, s1=40, s2=40, L = [1,1]):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.is_mesh = is_mesh
        self.s1 = s1
        self.s2 = s2

        self.fc0 = nn.Linear(in_channels, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, s1, s2, L = L)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2, s1, s2, L = L)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.b0 = nn.Conv2d(2, self.width, 1)
        self.b1 = nn.Conv2d(2, self.width, 1)
        self.b2 = nn.Conv2d(2, self.width, 1)
        self.b3 = nn.Conv2d(2, self.width, 1)
        self.b4 = nn.Conv1d(2, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

        # Optional model-side I/O normalization stats.
        # Shapes broadcast against u:(B,N,C).
        self.register_buffer("norm_u_in_mean", torch.zeros(1, 1, in_channels))
        self.register_buffer("norm_u_in_std", torch.ones(1, 1, in_channels))
        self.register_buffer("norm_u_out_mean", torch.zeros(1, 1, out_channels))
        self.register_buffer("norm_u_out_std", torch.ones(1, 1, out_channels))
        self.register_buffer("norm_enabled", torch.tensor(0, dtype=torch.uint8))

    def set_io_normalization(self, u_in_mean, u_in_std, u_out_mean, u_out_std, enabled=True, eps=1e-6):
        """Set model-side normalization statistics and toggle."""
        with torch.no_grad():
            in_mean = torch.as_tensor(u_in_mean, dtype=self.norm_u_in_mean.dtype, device=self.norm_u_in_mean.device).view(1, 1, -1)
            in_std = torch.as_tensor(u_in_std, dtype=self.norm_u_in_std.dtype, device=self.norm_u_in_std.device).view(1, 1, -1)
            out_mean = torch.as_tensor(u_out_mean, dtype=self.norm_u_out_mean.dtype, device=self.norm_u_out_mean.device).view(1, 1, -1)
            out_std = torch.as_tensor(u_out_std, dtype=self.norm_u_out_std.dtype, device=self.norm_u_out_std.device).view(1, 1, -1)

            self.norm_u_in_mean.copy_(in_mean)
            self.norm_u_in_std.copy_(torch.clamp(in_std, min=eps))
            self.norm_u_out_mean.copy_(out_mean)
            self.norm_u_out_std.copy_(torch.clamp(out_std, min=eps))
            self.norm_enabled.fill_(1 if enabled else 0)

    def set_io_normalization_enabled(self, enabled: bool):
        with torch.no_grad():
            self.norm_enabled.fill_(1 if enabled else 0)

    def forward(self, u, code=None, x_in=None, x_out=None, iphi=None):
        # u (batch, Nx, d) the input value
        # code (batch, Nx, d) the input features
        # x_in (batch, Nx, 2) the input mesh (sampling mesh)
        # xi (batch, xi1, xi2, 2) the computational mesh (uniform)
        # x_in (batch, Nx, 2) the input mesh (query mesh)

        if self.is_mesh and x_in == None:
            x_in = u
        if self.is_mesh and x_out == None:
            x_out = u
        if bool(self.norm_enabled.item()):
            u = (u - self.norm_u_in_mean) / self.norm_u_in_std
        grid = self.get_grid([u.shape[0], self.s1, self.s2], u.device).permute(0, 3, 1, 2)

        u = self.fc0(u)
        u = u.permute(0, 2, 1)

        uc1 = self.conv0(u, x_in=x_in, iphi=iphi, code=code)
        uc3 = self.b0(grid)
        uc = uc1 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv1(uc)
        uc2 = self.w1(uc)
        uc3 = self.b1(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv2(uc)
        uc2 = self.w2(uc)
        uc3 = self.b2(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        uc1 = self.conv3(uc)
        uc2 = self.w3(uc)
        uc3 = self.b3(grid)
        uc = uc1 + uc2 + uc3
        uc = F.gelu(uc)

        u = self.conv4(uc, x_out=x_out, iphi=iphi, code=code)
        u3 = self.b4(x_out.permute(0, 2, 1))
        u = u + u3

        u = u.permute(0, 2, 1)
        u = self.fc1(u)
        u = F.gelu(u)
        u = self.fc2(u)
        if bool(self.norm_enabled.item()):
            u = u * self.norm_u_out_std + self.norm_u_out_mean
        return u

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)



class IPHI(nn.Module):
    def __init__(self, width=32, device="cuda:0"):
        super().__init__()
        self.width = width

        # old raw feature dim was 4: [x, y, angle, radius]
        # new raw feature dim is 5: [x, y, sin(angle), cos(angle), radius]
        self.raw_dim = 5
        self.freq_dim = self.raw_dim * (self.width // 4)  # per sin/cos bank
        self.feat_dim = self.width + 2 * self.freq_dim    # fc0 + Fourier features

        self.fc0 = nn.Linear(self.raw_dim, self.width)
        self.fc_code = nn.Linear(42, self.width)

        # when code is absent
        self.fc_no_code = nn.Linear(self.feat_dim, 4 * self.width)
        # when code is present: concat(code_embed, feat)
        self.fc1 = nn.Linear(self.width + self.feat_dim, 4 * self.width)

        self.fc2 = nn.Linear(4 * self.width, 4 * self.width)
        self.fc3 = nn.Linear(4 * self.width, 2)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

        self.B = np.pi * torch.pow(
            2, torch.arange(0, self.width // 4, dtype=torch.float, device=device)
        ).reshape(1, 1, 1, self.width // 4)

    def forward(self, x, code=None):
        if code is not None:
            center = code[:, 0:2].unsqueeze(1)
        else:
            center = torch.tensor([0.5, 0.5], device=x.device, dtype=x.dtype).view(1, 1, 2).repeat(x.shape[0], 1, 1)

        dx = x[:, :, 0] - center[:, :, 0]
        dy = x[:, :, 1] - center[:, :, 1]
        angle = torch.atan2(dy, dx)
        radius = torch.norm(x - center, dim=-1, p=2)

        sin_a = torch.sin(angle)
        cos_a = torch.cos(angle)

        xd = torch.stack([x[:, :, 0], x[:, :, 1], sin_a, cos_a, radius], dim=-1)  # (b,n,5)

        b, n, d = xd.shape
        x_sin = torch.sin(self.B * xd.view(b, n, d, 1)).view(b, n, d * self.width // 4)
        x_cos = torch.cos(self.B * xd.view(b, n, d, 1)).view(b, n, d * self.width // 4)

        xd0 = self.fc0(xd)
        feat = torch.cat([xd0, x_sin, x_cos], dim=-1)  # (b,n,feat_dim)

        if code is not None:
            cd = self.fc_code(code).unsqueeze(1).repeat(1, n, 1)
            h = torch.cat([cd, feat], dim=-1)
        else:
            h = self.fc_no_code(feat)

        h = F.gelu(self.fc1(h))
        h = F.gelu(self.fc2(h))
        h = self.fc3(h)
        return x + h



def get_global_L_from_h5(h5_path: str, key: str | None = None):
    import h5py, numpy as np
    with h5py.File(h5_path, "r") as f:
        if key is None:
            keys = sorted([k for k in f.keys() if k.startswith("sample_")])
            key = keys[0]
        pos = f[key]["pos"][:]  # (N,2)
    L1 = float(pos[:, 0].max() - pos[:, 0].min())
    L2 = float(pos[:, 1].max() - pos[:, 1].min())
    return [L1, L2], key
