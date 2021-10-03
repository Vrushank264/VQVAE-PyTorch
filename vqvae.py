import torch
import torch.nn as nn
from torch.autograd import Function
from torchsummary import summary


def weight_init(model):
    
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(model.weight.data)
        model.bias.data.fill_(0)
        
class VectorQuantization(Function):
    
    @staticmethod
    def forward(ctx, inputs, codebook):
        """
        
        Parameters
        ----------
        ctx : context. It is used to store arbitrary data that can be retrieved
              during the backward pass.
        inputs : Tensor
            Encoded image vector(z_e_x).
        codebook : Tensor
            Embeddings.

        Returns
        -------
        idx : quantized indices / latents.

        """
        
        with torch.no_grad():
            
            input_size = inputs.size()
            emb_size = codebook.size(1)
            flat_input = inputs.view(-1, emb_size)
            
            codebook_sq = torch.sum(codebook ** 2, dim = 1)
            inputs_sq = torch.sum(inputs ** 2, dim = 1, keepdim = True)
            #torch.addmm: result = beta * input + alpha * (mat1[i] @ mat2[i]) 
            l2_dis = torch.addmm(input = codebook_sq + inputs_sq,
                                 mat1 = flat_input, mat2 = codebook.t(), 
                                 alpha = 2.0, beta = 1.0)
            _, idx_flatten = torch.min(l2_dis, dim = 1)
            idx = idx_flatten(*input_size[:-1])
            ctx.mark_non_differentiable(idx)
            
            return idx
    
    @staticmethod
    def backward(ctx, grad_outputs):
        
        raise RuntimeError('Trying to call backward on graph containing `Vector Quantization`'
                           'which is non-differentiable. Use VQStraightThrough instead.')
        

class VQStraightThrough(Function):
    
    @staticmethod
    def forward(ctx, inputs, codebook):
        
        idx = VQ(inputs, codebook)
        flat_idx = idx.view(-1)
        ctx.save_for_backward(flat_idx, codebook)
        ctx.mark_non_differentiable(flat_idx)
        codes_flatten = torch.index_select(input = codebook, dim = 0, index = flat_idx)
        codes = codes_flatten.view_as(inputs)
        
        return (codes, flat_idx)
    
    @staticmethod
    def backward(ctx, grad_outputs, grad_indices):
        
        grad_inputs, grad_codebook = None, None
        
        if ctx.needs_input_grad[0]:
            
            grad_inputs = grad_outputs.clone()
            
        if ctx.needs_input_grad[1]:
            
            idx, codebook = ctx.saved_tensors
            emb_size = codebook.size(1)
            flat_grad_output = (grad_outputs.contiguous().view(-1, emb_size))
            grad_codebook = torch.zeros_like(codebook)
            grad_codebook.index_add_(0, idx, flat_grad_output)
        
        return (grad_inputs, grad_codebook)
    
VQ = VectorQuantization.apply
VQ_ST =  VQStraightThrough.apply


class VQEmbedding(nn.Module):
    
    def __init__(self, K: int, D: int):
        
        """

        Parameters
        ----------
        K : int
            Total number of embeddings in codebook.
        D : int
            Dimensions of every embedding in the codebook.

        """
        
        super().__init__()
        self.vq_embs = nn.Embedding(K, D)
        self.vq_embs.weight.data.uniform_(-1.0 / K, 1.0 / K)
        
    def forward(self, z_e_x):
        
        z_e_x = z_e_x.permute(0,2,3,1).contiguous()
        latents = VQ(z_e_x, self.vq_embs.weight)
        return latents
    
    def straight_through_forward(self, z_e_x):
        
        zex = z_e_x.permute(0,2,3,1).contiguous()
        z_q_x, idx = VQ_ST(zex, self.vq_embs.weight.detach())
        z_q_x = z_q_x.permute(0,3,1,2).contiguous()
        flat_zqx_tilde = torch.index_select(self.vq_embs.weight, dim = 0, index = idx)
        zqx_tilde = flat_zqx_tilde.view_as(zex)
        zqx_tilde = zqx_tilde.permute(0,3,1,2).contiguous()
        return z_q_x, zqx_tilde


class ResidualBlock(nn.Module):
    
    def __init__(self, channels):
        
        super().__init__()
        self.resblock = nn.Sequential(nn.ReLU(inplace = True),
                                      nn.Conv2d(channels, channels, kernel_size = 3, stride = 1, padding = 1),
                                      nn.BatchNorm2d(channels),
                                      nn.ReLU(inplace = True),
                                      nn.Conv2d(channels, channels, 1),
                                      nn.BatchNorm2d(channels)
                                      )
        
    def forward(self, x):
        
        return self.resblock(x) + x
    

class VQVAE(nn.Module):
    
    def __init__(self, in_c, out_c, K = 512):
        
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(in_c, out_c, kernel_size = 4, stride = 2, padding = 1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU(inplace = True),
                                     nn.Conv2d(out_c, out_c, kernel_size = 4, stride = 2, padding = 1),
                                     ResidualBlock(out_c),
                                     ResidualBlock(out_c)
                                     )
        self.codebook = VQEmbedding(K, D = out_c)
        
        self.decoder = nn.Sequential(ResidualBlock(out_c),
                                     ResidualBlock(out_c),
                                     nn.ReLU(inplace = True),
                                     nn.ConvTranspose2d(out_c, out_c, kernel_size = 4, stride = 2, padding = 1),
                                     nn.BatchNorm2d(out_c),
                                     nn.ReLU(inplace = True),
                                     nn.ConvTranspose2d(out_c, in_c, kernel_size = 4, stride = 2, padding = 1),
                                     nn.Tanh()
                                     )
        
        self.apply(weight_init)
        
    def encode(self, x):
        
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents
    
    def decode(self, latents):
        
        z_q_x = self.codebook.vq_embs(latents).permute(0,3,1,2)
        x_tilde = self.decoder(z_q_x)
        return x_tilde
    
    def forward(self, x):
        
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through_forward(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return z_e_x, z_q_x, x_tilde

    
if __name__ == '__main__':
    
    vqvae = VQVAE(3, 256).to(torch.device('cuda'))
    print(summary(vqvae, (3,32,32))) 