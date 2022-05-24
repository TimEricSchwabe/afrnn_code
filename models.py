import torch
import torch.nn.functional as F
import torch.nn as nn

# AFRNN
class FRNN_AS_SC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, epsilon=0.1, gamma=0.15):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.gamma = gamma
        
        self.Eye = torch.eye(self.hidden_size, self.hidden_size).to(device)
        
        #Input to Hidden Weight:
        ith_weights = torch.Tensor(hidden_size, input_size)
        self.ith_weights = nn.Parameter(ith_weights, requires_grad=True)
        
        ith_bias = torch.Tensor(hidden_size)
        self.ith_bias = nn.Parameter(ith_bias, requires_grad=True)
        
        #Hidden to Hidden Weight:
        hth_weights = torch.Tensor(hidden_size, hidden_size)
        self.hth_weights = nn.Parameter(hth_weights,requires_grad=True)
        
        #Hidden2 to Hidden2 Weights:
        h2th2_weights = torch.Tensor(hidden_size, hidden_size)
        self.h2th2_weights = nn.Parameter(h2th2_weights,requires_grad=True)
        
        h2_bias = torch.Tensor(hidden_size)
        self.h2_bias = nn.Parameter(h2_bias, requires_grad=True)
        
        #Feedback Weights:
        fb_weights = torch.Tensor(hidden_size, hidden_size)
        self.fb_weights = nn.Parameter(fb_weights,requires_grad=True)
        
        #Hidden to Output Weights:
        hto_weights = torch.Tensor(output_size, hidden_size)
        self.hto_weights = nn.Parameter(hto_weights,requires_grad=True)
        
        o_bias = torch.Tensor(output_size)
        self.o_bias = nn.Parameter(o_bias, requires_grad=True)
        
        #Init weights
        nn.init.normal_(self.ith_weights, mean=0, std=1);
        nn.init.normal_(self.hto_weights, mean=0, std=1/hidden_size);
        nn.init.normal_(self.hth_weights, mean=0, std=1/hidden_size);
        #nn.init.normal_(self.hth2_weights, mean=0, std=1/hidden_size);
        nn.init.normal_(self.h2th2_weights, mean=0, std=1/hidden_size);
        nn.init.normal_(self.fb_weights, mean=0, std=1/output_size);
    
        #Init biases
        nn.init.zeros_(self.ith_bias)
        nn.init.zeros_(self.h2_bias)
        nn.init.zeros_(self.o_bias)


           
        
    def forward(self, input, hidden_old, hidden2_old):
        #Enforce antisymmetric weights
        hth = self.hth_weights - self.hth_weights.t()

        hth = hth - self.gamma * self.Eye
        fb = self.fb_weights
        fb = fb - self.gamma * self.Eye
        h2th2 = self.h2th2_weights - self.h2th2_weights.t()

        h2th2 = h2th2 - self.gamma * self.Eye

        
        #Calculate bottom_layer
        hidden = torch.tanh(torch.mm(input, self.ith_weights.t())
                           +torch.mm(hidden_old, hth.t())
                           +torch.mm(hidden2_old, -fb.t())
                           +self.ith_bias)
        #Skip Connection:
        hidden = hidden_old +  self.epsilon * hidden
        
        #Calculate Top-Layer:
        hidden2 = torch.tanh(torch.mm(hidden, fb)
                            +torch.mm(hidden2_old, h2th2.t())
                            +self.h2_bias)

        
        hidden2 = hidden2_old + self.epsilon *  hidden2
    
        
        output = torch.mm(hidden2, self.hto_weights.t()) + self.o_bias

        return output, hidden, hidden2

# FRNN
class FRNN_SC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, epsilon=0.1, gamma=0.15):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.gamma = gamma
        
        self.Eye = torch.eye(self.hidden_size, self.hidden_size).to(device)
        
        #Input to Hidden Weight:
        ith_weights = torch.Tensor(hidden_size, input_size)
        self.ith_weights = nn.Parameter(ith_weights, requires_grad=True)
        
        ith_bias = torch.Tensor(hidden_size)
        self.ith_bias = nn.Parameter(ith_bias, requires_grad=True)
        
        #Hidden to Hidden Weight:
        hth_weights = torch.Tensor(hidden_size, hidden_size)
        self.hth_weights = nn.Parameter(hth_weights,requires_grad=True)
        
        
        #Hidden to Hidden2 Weights:
        hth2_weights = torch.Tensor(hidden_size, hidden_size)
        self.hth2_weights = nn.Parameter(hth2_weights,requires_grad=True)
        
        #Hidden2 to Hidden2 Weights:
        h2th2_weights = torch.Tensor(hidden_size, hidden_size)
        self.h2th2_weights = nn.Parameter(h2th2_weights,requires_grad=True)
        
        h2_bias = torch.Tensor(hidden_size)
        self.h2_bias = nn.Parameter(h2_bias, requires_grad=True)
        
        #Feedback Weights:
        fb_weights = torch.Tensor(hidden_size, hidden_size)
        self.fb_weights = nn.Parameter(fb_weights,requires_grad=True)
        
        #Hidden to Output Weights:
        hto_weights = torch.Tensor(output_size, hidden_size)
        self.hto_weights = nn.Parameter(hto_weights,requires_grad=True)
        
        o_bias = torch.Tensor(output_size)
        self.o_bias = nn.Parameter(o_bias, requires_grad=True)
        
        #Init weights
        nn.init.normal_(self.ith_weights, mean=0, std=1);
        nn.init.normal_(self.hto_weights, mean=0, std=1/hidden_size);
        nn.init.normal_(self.hth_weights, mean=0, std=1/hidden_size);
        nn.init.normal_(self.hth2_weights, mean=0, std=1/hidden_size);
        nn.init.normal_(self.h2th2_weights, mean=0, std=1/hidden_size);
        nn.init.normal_(self.fb_weights, mean=0, std=1/output_size);
    
        #Init biases
        nn.init.zeros_(self.ith_bias)
        nn.init.zeros_(self.h2_bias)
        nn.init.zeros_(self.o_bias)


        
        
    def forward(self, input, hidden_old, hidden2_old):
        #Enforce antisymmetric weights
        hth = self.hth_weights - self.hth_weights.t()
        #hth = self.hth_weights

        hth = hth - self.gamma * self.Eye
        #fb = self.fb_weights - self.fb_weights.t()
        hth2 = self.hth2_weights - self.gamma *self.Eye
        fb = self.fb_weights
        fb = fb - self.gamma * self.Eye
        h2th2 = self.h2th2_weights - self.h2th2_weights.t()
        #h2th2 = self.h2th2_weights

        h2th2 = h2th2 - self.gamma * self.Eye

        
        #Calculate bottom_layer
        hidden = torch.tanh(torch.mm(input, self.ith_weights.t())
                           +torch.mm(hidden_old, hth.t())
                           +torch.mm(hidden2_old, -fb.t())
                           +self.ith_bias)
        #Skip Connection:
        hidden = hidden_old +  self.epsilon * hidden
        
        #Calculate Top-Layer:
        hidden2 = torch.tanh(torch.mm(hidden, hth2.t())
                            +torch.mm(hidden2_old, h2th2.t())
                            +self.h2_bias)

        
        hidden2 = hidden2_old + self.epsilon *  hidden2
    
        
        output = torch.mm(hidden2, self.hto_weights.t()) + self.o_bias
        return output, hidden, hidden2

    
# 2-ARNN
class TLRNN_AS_SC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, epsilon=0.1, gamma=0.15):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.gamma = gamma
        
        self.Eye = torch.eye(self.hidden_size, self.hidden_size).to(device)
        
        #Input to Hidden Weight:
        ith_weights = torch.Tensor(hidden_size, input_size)
        self.ith_weights = nn.Parameter(ith_weights, requires_grad=True)
        
        ith_bias = torch.Tensor(hidden_size)
        self.ith_bias = nn.Parameter(ith_bias, requires_grad=True)
        
        #Hidden to Hidden Weight:
        hth_weights = torch.Tensor(hidden_size, hidden_size)
        self.hth_weights = nn.Parameter(hth_weights,requires_grad=True)
        
        
        #Hidden to Hidden2 Weights:
        hth2_weights = torch.Tensor(hidden_size, hidden_size)
        self.hth2_weights = nn.Parameter(hth2_weights,requires_grad=True)
        
        #Hidden2 to Hidden2 Weights:
        h2th2_weights = torch.Tensor(hidden_size, hidden_size)
        self.h2th2_weights = nn.Parameter(h2th2_weights,requires_grad=True)
        
        h2_bias = torch.Tensor(hidden_size)
        self.h2_bias = nn.Parameter(h2_bias, requires_grad=True)
        
        #Feedback Weights:
        fb_weights = torch.Tensor(hidden_size, hidden_size)
        self.fb_weights = nn.Parameter(fb_weights,requires_grad=True)
        
        #Hidden to Output Weights:
        hto_weights = torch.Tensor(output_size, hidden_size)
        self.hto_weights = nn.Parameter(hto_weights,requires_grad=True)
        
        o_bias = torch.Tensor(output_size)
        self.o_bias = nn.Parameter(o_bias, requires_grad=True)
        
        #Init weights
        nn.init.normal_(self.ith_weights, mean=0, std=1);
        nn.init.normal_(self.hto_weights, mean=0, std=1/hidden_size);
        nn.init.normal_(self.hth_weights, mean=0, std=1/hidden_size);
        nn.init.normal_(self.hth2_weights, mean=0, std=1/hidden_size);
        nn.init.normal_(self.h2th2_weights, mean=0, std=1/hidden_size);
        nn.init.normal_(self.fb_weights, mean=0, std=1/output_size);
    
        #Init biases
        nn.init.zeros_(self.ith_bias)
        nn.init.zeros_(self.h2_bias)
        nn.init.zeros_(self.o_bias)


        
        
    def forward(self, input, hidden_old, hidden2_old):
        #Enforce antisymmetric weights
        hth = self.hth_weights - self.hth_weights.t()

        hth = hth - self.gamma * self.Eye
        fb = self.fb_weights
        fb = fb - self.gamma * self.Eye
        h2th2 = self.h2th2_weights - self.h2th2_weights.t()

        h2th2 = h2th2 - self.gamma * self.Eye

        
        #Calculate bottom_layer
        hidden = torch.tanh(torch.mm(input, self.ith_weights.t())
                           +torch.mm(hidden_old, hth.t())
                           +self.ith_bias)
        #Skip Connection:
        hidden = hidden_old +  self.epsilon * hidden
        
        #Calculate Top-Layer:
        hidden2 = torch.tanh(torch.mm(hidden, fb)
                            +torch.mm(hidden2_old, h2th2.t())
                            +self.h2_bias)
        
        hidden2 = hidden2_old + self.epsilon *  hidden2
    
        
        output = torch.mm(hidden2, self.hto_weights.t()) + self.o_bias
        return output, hidden, hidden2


