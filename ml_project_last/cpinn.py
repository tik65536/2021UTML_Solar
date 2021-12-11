import torch
from dnn import *
from cdnn import *
import numpy as np

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)


class CPINN():
    
    def __init__(self, x,x_v,l,l_v,t,t_v,y,y_v,flux,vflux,k_size,s_layers,pgf_layers,flux_layers,bsize,tb=None):
        
        self.device=torch.device('cpu')
        #if torch.cuda.is_available():
        #    self.device = torch.device('cuda')
        #else:
        #    self.device = torch.device('cpu')
            
        # Training data
        self.tb=tb
        self.x = torch.tensor(x[:,0:5,:,:], requires_grad=True).float().to(self.device)
        self.l = torch.tensor(l, requires_grad=True).float().to(self.device)
        self.x_t = torch.tensor(x[:,5:6,:,:], requires_grad=True).float().to(self.device)
        self.u = torch.tensor(y[:,0].reshape(-1,1)).float().to(self.device)
        self.u2 = torch.tensor(y[:,1].reshape(-1,1)).float().to(self.device)
        self.tt =  torch.tensor(t, requires_grad=True).float().to(self.device)
        self.flux = torch.tensor(flux,requires_grad=True).float().to(self.device)
    
        # Validation Data
        self.x_v = torch.tensor(x_v[:,0:5,:,:], requires_grad=False).float()
        self.l_v = torch.tensor(l_v, requires_grad=False).float()
        self.x_t_v = torch.tensor(x_v[:,5:6,:,:], requires_grad=False).float()
        self.u_v = torch.tensor(y_v[:,0].reshape(-1,1),requires_grad=False).float()
        self.u2_v = torch.tensor(y_v[:,1].reshape(-1,1),requires_grad=False).float()
        self.tt_v =  torch.tensor(t_v, requires_grad=False).float()
        self.vflux = torch.tensor(vflux,requires_grad=False).float()
        
        # deep neural networks for U
        self.birth_dnn = CDNN(k_size,x.shape[3],x.shape[2],x.shape[1]).to(self.device)
        self.death_dnn = CDNN(k_size,x.shape[3],x.shape[2],x.shape[1]).to(self.device)
        self.s_dnn = DNN(s_layers).to(self.device)
        self.pgf_dnn = DNN(pgf_layers).to(self.device)
        self.pmf_dnn = DNN(flux_layers).to(self.device)
        #self.flux_dnn = DNN(flux_layers).to(self.device)
        
        self.birth_dnn.apply(weights_init)
        self.death_dnn.apply(weights_init)
        self.s_dnn.apply(weights_init)
        self.pgf_dnn.apply(weights_init)
        self.pmf_dnn.apply(weights_init)
        
        self.parameters = set()
        self.parameters |= set(self.birth_dnn.parameters())
        self.parameters |= set(self.death_dnn.parameters())
        self.parameters |= set(self.s_dnn.parameters())
        self.parameters |= set(self.pgf_dnn.parameters())
        self.parameters |= set(self.pmf_dnn.parameters())
        
        #self.optimizer_AdamFlux = torch.optim.Adam(self.flux_dnn.parameters())
        self.optimizer_Adam = torch.optim.Adam(self.parameters)
        self.iter = 0
        self.bsize = bsize
        self.bestValidationLoss=float('inf')
                  
    
    def net_birth(self, x, t):  
        u = self.birth_dnn(torch.cat([x, t], dim=1))
        return u
    
    def net_death(self, x, t):  
        u = self.death_dnn(torch.cat([x, t], dim=1))
        return u

    def net_s(self,b,d,l,flux,t):  
        #print(x.size())
        #print(t.size())
        u = self.s_dnn(torch.cat([b,d,l,flux,t], dim=1))
        return u
    
    def net_pgf(self, s, t):  
        u = self.pgf_dnn(torch.cat([s,t], dim=1))
        return u

    def net_pmf(self,s,f,t):  
        u = self.pmf_dnn(torch.cat([s,f,t], dim=1))
        return u

    def net_flux(self,s,flux,t):  
        u = self.flux_dnn(torch.cat([s,flux,t], dim=1))
        return u
    
    def net_f(self,x,l,flux,t,tt):
        """ The pytorch autograd version of calculating residual """
        #print(torch.median(t,dim=2))
        #print(x)
        b = self.net_birth(x, t)
        d = self.net_death(x, t)
        s = self.net_s(b,d,l,flux, tt)
        u = self.net_pgf(s, tt)
        #pmf = self.net_pmf(s, tt)
        
        u_t = torch.autograd.grad(
            u, tt, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_s = torch.autograd.grad(
            u, s, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        f = (1-(s))*(d-b*(s))*(u_s) - u_t
        
        return f
        
    def train(self, nIter):
        
        self.birth_dnn.train()
        self.death_dnn.train()
        self.s_dnn.train()
        self.pgf_dnn.train()
        self.pmf_dnn.train()
        
        batch=self.x.shape[0]//self.bsize
        validation_loss=0
    
        for epoch in range(nIter):
            self.birth_dnn.train()
            self.death_dnn.train()
            self.s_dnn.train()
            self.pgf_dnn.train()
            self.pmf_dnn.train()
            running_loss=0
            u_running_loss=0
            f_running_loss=0
            for i in range(batch):
                idx=np.arange(i*self.bsize,i*self.bsize+self.bsize,dtype=np.int64)
                # using traing data
                b_pred = self.net_birth(self.x[idx,:,:,:], self.x_t[idx,:,:])
                d_pred = self.net_death(self.x[idx,:,:], self.x_t[idx,:,:])
                s_pred = self.net_s(b_pred,d_pred,self.l[idx],self.flux[idx],self.tt[idx])
                
                m_pred = self.net_pmf(s_pred,self.flux[idx],self.tt[idx])
                
                f_pred = self.net_f(self.x[idx,:,:,:], self.l[idx],self.flux[idx],self.x_t[idx,:,:],self.tt[idx])
                loss_u = torch.mean(torch.abs(self.u2[idx] - m_pred))
                #loss_flux = torch.mean((self.u2[idx] - flux_pred) ** 2)
                loss_f = torch.mean(f_pred ** 2)
                avg_b_pred = torch.mean(b_pred)
                avg_d_pred = torch.mean(d_pred)
                median_mpred = torch.median(m_pred)
                q25_m = torch.quantile(m_pred,0.25)
                q75_m = torch.quantile(m_pred,0.75)
                loss = loss_u+loss_f
                # Backward and optimize
                self.optimizer_Adam.zero_grad()
                loss.backward()
                self.optimizer_Adam.step()
                print('Batch: %3d, Loss_u: %.5e, Loss_f: %.5e,avg_b: %.5e, avg_d: %.5e, quantile25_m: %.5e, median_m: %.5e, quantile75_m: %.5e\r' % (i, loss_u.item(),loss_f.item(),avg_b_pred,avg_d_pred,q25_m,median_mpred,q75_m),end='')
                running_loss += loss.item()
                u_running_loss += loss_u.item()
                f_running_loss += loss_f.item()
            
            ##Call for validation
            validu=self.validate(self.tb,epoch)
            validation_loss = validu
            if(validation_loss<self.bestValidationLoss):
                torch.save({
                            'birth_dnn': self.birth_dnn.state_dict(),
                            'death_dnn': self.death_dnn.state_dict(),
                            's_dnn': self.s_dnn.state_dict(),
                            'pgf_dnn': self.pgf_dnn.state_dict(),
                            'pmf_dnn': self.pmf_dnn.state_dict(),
                            'optimizer': self.optimizer_Adam.state_dict()
                            }, './CPINN_Models/CPINN_Integrate2PGF_'+str(epoch)+'_vloss'+str(validation_loss))
                self.bestValidationLoss=validation_loss
                
            if(self.tb!=None):
                self.tb.add_scalars("Training Batch Loss",{'Avg Loss':running_loss/batch,'Avg U Loss':u_running_loss/batch,'Avg F Loss':f_running_loss/batch} , epoch)
                self.tb.add_scalar("Validation batch flux Loss", validation_loss, epoch)
                for name, weight in self.birth_dnn.dnn1.named_parameters():
                    self.tb.add_histogram('birth_'+name,weight, epoch)
                    self.tb.add_histogram(f'birth_{name}.grad',weight.grad, epoch)
                for name, weight in self.death_dnn.dnn1.named_parameters():
                    self.tb.add_histogram('death_'+name,weight, epoch)
                    self.tb.add_histogram(f'death_{name}.grad',weight.grad, epoch)
                for name, weight in self.s_dnn.layers.named_parameters():
                    self.tb.add_histogram('s_'+name,weight, epoch)
                    self.tb.add_histogram(f's_{name}.grad',weight.grad, epoch)
                for name, weight in self.pmf_dnn.layers.named_parameters():
                    self.tb.add_histogram('pmf_'+name,weight, epoch)
                    self.tb.add_histogram(f'pmf_{name}.grad',weight.grad, epoch)
                for name, weight in self.pgf_dnn.layers.named_parameters():
                    self.tb.add_histogram('pgf_'+name,weight, epoch)
                    #self.tb.add_histogram(f'pgf_{name}.grad',weight.grad, epoch)
                #for name, weight in self.flux_dnn.layers.named_parameters():
                #    self.tb.add_histogram('flux_'+name,weight, epoch)
                #    self.tb.add_histogram(f'flux_{name}.grad',weight.grad, epoch)
                
            print('\n\tEpoch: %d, Tarining Avg Loss: %.6e, Avg U Loss: %.6e, Avg F loss: %.6e' % (epoch, running_loss/batch,u_running_loss/batch,f_running_loss/batch))
            print('\tEpoch: %d, Validation Avg Loss: %.6e, Avg U Loss: %.6e \n' % (epoch, validation_loss,validu))
        torch.save({
                        'birth_dnn': self.birth_dnn.state_dict(),
                        'death_dnn': self.death_dnn.state_dict(),
                        's_dnn': self.s_dnn.state_dict(),
                            'pgf_dnn': self.pgf_dnn.state_dict(),
                            'pmf_dnn': self.pmf_dnn.state_dict(),
                            'optimizer': self.optimizer_Adam.state_dict()
        }, './CPINN_Models/CPINN_Integrate2PGF_'+str(epoch)+'_vloss'+str(u_running_loss/batch))   
            
    def validate(self,tb,epoch):
        self.birth_dnn.eval()
        self.death_dnn.eval()
        self.s_dnn.eval()
        self.pgf_dnn.eval()
        self.pmf_dnn.eval()
        with torch.no_grad():
            idx=np.random.randint(0,self.x_v.shape[0],self.bsize)
            ## Should using validation data
            x=self.x_v[idx,:,:,:].to(self.device)
            l=self.l_v[idx].to(self.device)
            x_t=self.x_t_v[idx,:,:].to(self.device)
            tt=self.tt_v[idx].to(self.device)
            u = self.u_v[idx].to(self.device)
            u2 = self.u2_v[idx].to(self.device)
            flux = self.vflux[idx].to(self.device)
            #Inference
            b_pred = self.net_birth(x,x_t)
            d_pred = self.net_death(x,x_t)
            s_pred = self.net_s(b_pred,d_pred,l,flux,tt)
            m_pred = self.net_pmf(s_pred,flux,tt)
            #Statistic
            avg_b_pred = torch.mean(b_pred)
            avg_d_pred = torch.mean(d_pred)
            median_mpred = torch.median(m_pred)
            q25_m = torch.quantile(m_pred,0.25)
            q75_m = torch.quantile(m_pred,0.75)
        #f_pred = self.net_f(x, l,flux,x_t,tt)
        #Loss
        loss_u = torch.mean(torch.abs(u2 - m_pred))
        # Ensure the gradient is not calculated
        #loss_f = torch.mean(f_pred ** 2)
            #Logging
        if(tb!=None): 
            tb.add_histogram('Validation CPINN Integration2PGF Predict',m_pred, epoch)
            tb.add_histogram('Validation CPINN Integration2PGF Actual',u2, epoch)
            tb.add_pr_curve('Validation PR Curve',u2,m_pred,global_step=epoch)
            tb.add_scalars("Validation CPINN Integration2PGF Loss", {'Avg U Loss':loss_u}, epoch)
            tb.add_scalars('Validation CPINN Integration2PGF Avg B D',{'Avg B':avg_b_pred,'Avg D':avg_d_pred}, epoch)
            tb.add_scalars('Validation CPINN Integration2PGF flux statistic',{'median':median_mpred,'q25':q25_m,'q75':q75_m},epoch)
        return loss_u 
    
    def predict(self,x,l,x_t,tt,flux):
        self.birth_dnn.eval()
        self.death_dnn.eval()
        self.s_dnn.eval()
        self.pgf_dnn.eval()
        self.pmf_dnn.eval()
        with torch.no_grad():
            x = torch.tensor(x).float().to(self.device)
            l = torch.tensor(l).float().to(self.device)
            x_t = torch.tensor(x_t).float().to(self.device)
            tt=torch.tensor(tt).float().to(self.device)
            flux = torch.tensor(flux).float().to(self.device)
            b_pred = self.net_birth(x,x_t)
            d_pred = self.net_death(x,x_t)
            s_pred = self.net_s(b_pred,d_pred,l,flux,tt)
            m_pred = self.net_pmf(s_pred,flux,tt)
            print(f'Predict Flux : {torch.reshape(m_pred,(1,-1))}')
        return m_pred
    
    def saveMode(self,name):
        torch.save(self,'./'+name)
    
    def loadModel(self,PATH):
        checkpoint = torch.load(PATH)
        self.birth_dnn.load_state_dict(checkpoint['birth_dnn'])
        self.death_dnn.load_state_dict(checkpoint['death_dnn'])
        self.s_dnn.load_state_dict(checkpoint['s_dnn'])
        self.pgf_dnn.load_state_dict(checkpoint['pgf_dnn'])
        self.pmf_dnn.load_state_dict(checkpoint['pmf_dnn'])
        self.optimizer_Adam.load_state_dict(checkpoint['optimizer'])
        self.birth_dnn.to(self.device)
        self.death_dnn.to(self.device)
        self.s_dnn.to(self.device)
        self.pgf_dnn.to(self.device)
        self.pmf_dnn.to(self.device)
        self.birth_dnn.eval()
        self.death_dnn.eval()
        self.s_dnn.eval()
        self.pgf_dnn.eval()
        self.pmf_dnn.eval()
        