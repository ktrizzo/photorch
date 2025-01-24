import torch.nn as nn
import torch

class allparameters(nn.Module):
    def __init__(self):
        super(allparameters, self).__init__()

        self.Ca = torch.tensor(420.0)

class gsACi(nn.Module):
    def __init__(self,gs):
        super(gsACi, self).__init__()
        self.Ci = nn.Parameter(torch.ones(len(gs))*300)
        self.Ca = torch.tensor(420.0)
        self.gs = gs
    def forward(self):
        An = self.gs*(self.Ca - self.Ci)
        return An

class lossA(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, An_fvcb, An_gs,Ci):
        loss = self.mse(An_fvcb, An_gs)
        loss += torch.sum(torch.relu(-Ci))*100
        return loss

# Ball Woodrow Berry
class BWB(nn.Module):
    def __init__(self, An, rh, lcpd = None):
        super(BWB, self).__init__()
        if lcpd is None:
            self.num_FGs = 1
            self.FGs = torch.tensor([0])
        else:
            self.num_FGs = lcpd.num_FGs
            self.FGs = lcpd.FGs
        self.Ca = torch.tensor(420.0)
        self.A = An
        self.rh = rh
        self.gs0 = nn.Parameter(torch.ones(self.num_FGs))
        self.a1 = nn.Parameter(torch.ones(self.num_FGs))
    def forward(self):
        gs0 = self.gs0[self.FGs]
        a1 = self.a1[self.FGs]
        gs = gs0 + a1 * self.A * self.rh / self.Ca
        return gs

# Ball Berry Leuning
class BBL(nn.Module):
    def __init__(self, An, Gamma, VPD, lcpd = None):
        super(BBL, self).__init__()
        if lcpd is None:
            self.num_FGs = 1
            self.FGs = torch.tensor([0])
        else:
            self.num_FGs = lcpd.num_FGs
            self.FGs = lcpd.FGs
        self.Gamma = Gamma
        self.VPD = VPD
        self.A = An
        self.gs0 = nn.Parameter(torch.ones(self.num_FGs))
        self.a1 = nn.Parameter(torch.ones(self.num_FGs))
        self.D0 = nn.Parameter(torch.ones(self.num_FGs))
        self.Ca = torch.tensor(420.0)
    def forward(self):
        gs0 = self.gs0[self.FGs]
        a1 = self.a1[self.FGs]
        D0 = self.D0[self.FGs]
        gs = gs0 + a1 * self.A / (self.Ca - self.Gamma) / (1 + self.VPD / D0)
        return gs

# Medlyn et al.
class MED(nn.Module):
    def __init__(self, An, VPD, lcpd = None):
        super(MED, self).__init__()
        if lcpd is None:
            self.num_FGs = 1
            self.FGs = torch.tensor([0])
        else:
            self.num_FGs = lcpd.num_FGs
            self.FGs = lcpd.FGs
        self.A = An
        self.VPD = VPD
        self.gs0 = nn.Parameter(torch.ones(self.num_FGs))
        self.g1 = nn.Parameter(torch.ones(self.num_FGs))
        self.Ca = torch.tensor(420.0)
    def forward(self):
        gs0 = self.gs0[self.FGs]
        g1 = self.g1[self.FGs]
        gs = gs0 + 1.6 * (1 + g1 / torch.sqrt(self.VPD / 1000 * 101.3)) * self.A / self.Ca
        return gs

# Buckley Mott Farquhar
class BMF(nn.Module):
    def __init__(self, Q, VPD,lcpd=None):
        super(BMF, self).__init__()
        if lcpd is None:
            self.num_FGs = 1
            self.FGs = torch.tensor([0])
        else:
            self.num_FGs = lcpd.num_FGs
            self.FGs = lcpd.FGs
        self.Q = Q
        self.VPD = VPD
        self.Em = nn.Parameter(torch.ones(self.num_FGs))
        self.i0 = nn.Parameter(torch.ones(self.num_FGs)*10)
        self.k = nn.Parameter(torch.ones(self.num_FGs)*10000)
        self.b = nn.Parameter(torch.ones(self.num_FGs)*10)
    def forward(self):
        Em = self.Em[self.FGs]
        i0 = self.i0[self.FGs]
        k = self.k[self.FGs]
        b = self.b[self.FGs]
        gs = Em * (self.Q + i0) / (k + b * self.Q + (self.Q + i0) * self.VPD)
        return gs

class lossSC(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, scm,gs_fit,gs_true):
        loss = self.mse(gs_fit, gs_true)
        # get all learnable parameters in scm
        for param in scm.parameters():
            loss += torch.sum(torch.relu(-param))*10
        return loss
