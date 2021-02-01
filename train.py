from model import Transformer,Generator,Discriminator
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import grad
from argumentparser import ArgumentParser
from data_processing import train_loader
from utils import bit_entropy
arg = ArgumentParser()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
D1 = Discriminator().to(device)
G1 = Generator().to(device)
model = Transformer(arg.num_size).to(device)
loss_fn = nn.MSELoss().to(device)


optimizer = optim.Adam(model.parameters(),lr=arg.lr)
optimizer_d = optim.Adam(D1.parameters(),lr=arg.lr)
optimizer_g = optim.Adam(G1.parameters(),lr=arg.lr)

def train(data_iter, model,D,G):

    for i in range(arg.num_epoch):
        model.train()
        D1.train()
        G1.train()
        for it, data in enumerate(data_iter):
            data = data.to(device)
            out,input1, binary_code, out1 = model.forward(data)
            out1 = torch.reshape(out1, (arg.batch_size, -1, arg.window, arg.code_size))
            real_data, real_code, _ = D(out1)
            inputs = G(input1)
            optimizer_d.zero_grad()
            out1 = torch.reshape(out1,(arg.batch_size,-1,arg.window,arg.code_size))
            real_data ,real_code, _=D(out1)
            min_Loss = bit_entropy(real_code)
            alpha = torch.randn((arg.batch_size,1,1,1)).to(device)
            x_hat = alpha*out1+(1-alpha)*inputs
            pred_hat,_,_ = D(x_hat)
            gradiants = grad(outputs=pred_hat,inputs=x_hat,grad_outputs=torch.ones(pred_hat.size()).to(device),
                             create_graph=True,retain_graph=True,only_inputs=True)[0]
            gradiant_penalty = arg.la*((gradiants.view(gradiants.size()[0],-1).norm(2,1)-1)**2).mean()
            D_res1,_,_ = D(inputs)
            D_res2,_,_ = D(out1)
            D_loss1 = -torch.mean(D_res2)+torch.mean(D_res1)+gradiant_penalty
            optimizer.zero_grad()
            loss = loss_fn(out, data)+D_loss1*arg.lambda_1+min_Loss*arg.lambda_2
            loss.backward(retain_graph=True)
            optimizer_g.zero_grad()
            normal_noise = torch.randn(arg.batch_size,arg.window,arg.code_size).normal_(0, 1).to(device)
            out1 = out1.squeeze()
            output2 = torch.cat((normal_noise, out1), 2)
            fake_images = G(output2)
            outputs,_,_ = D(fake_images)
            g_loss = -torch.mean(outputs)
            g_loss.backward(retain_graph=True)
            optimizer_g.step()
            optimizer.step()
            optimizer_d.step()
            if i % 20 == 0:
                torch.save(model.state_dict(), './model/model_trans_gan_64_air_{}.pkl'.format(i))
            if it%1000 == 0:
                print('iteration:', i, ',', it, 'loss:', loss.item(), 'loss_m:', min_Loss.item(), 'loss_d:',
                      D_loss1.item())

if __name__ == "__main__":
    train(train_loader,model,D1,G1)
