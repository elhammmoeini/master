import torch, pathlib, torchvision, os, random, glob, shutil, cv2, sys
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

base_path = pathlib.Path(__file__).parent.absolute()
sys.path.insert(0, base_path.as_posix())
import lrp

from lrp.patterns import fit_patternnet, fit_patternnet_positive # PatternNet patterns
from utils import store_patterns, load_patterns
from visualization import project, clip_quantile, heatmap_grid, grid
from torchcam.utils import overlay_mask
import torchcam
from tqdm import tqdm
from PIL import Image
from matplotlib.pyplot import imshow
from torchvision.transforms.functional import to_pil_image
from torch import nn
from .utils import CustomDataSet,ToCUDA,clone_weights,weights_checker\
                  ,PrepareData
from torch.utils.tensorboard import SummaryWriter

class AddLayer(nn.Module):
    
    def __init__(self, model, extra_layer):
        super().__init__()
        self.model=model
        self.extra_layer=extra_layer

    def forward(self,x):
        return self.extra_layer(self.model(x))

class main():

    def __init__(self, configs, state="train"):
        self.configs=configs
        self.device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        if state=="validation":
            self.camapp={}
            self.models={}
            if self.configs.MODE=="LRP":
                self.model=self.change_state(state, self.define_and_load_model())
                self.xai_mode=self.configs.LRP_RULE
            elif self.configs.MODE=="CAM":
                self.xai_mode=self.configs.CAM
                for cam_method in self.configs.CAM:
                    self.camapp[cam_method]=getattr(torchcam.methods, cam_method)
                    self.models[cam_method]=self.change_state(state, self.define_and_load_model())
        if state=="middle_layer":
            self.model=self.change_state(state, self.define_and_load_model())
            self.activation={}
            self.RegisterHook()
        else:
            self.writer=SummaryWriter("runs/classification")

        self.DesiredAccuracy=self.configs.DESIRED_ACCURACY
        self.softmax=self.on_cuda(nn.Softmax(dim=1))

    def define_and_load_model(self):
        self.model_zoo=torchvision.models
        print(f"Loading model : { self.configs.MODEL}")

        if  self.configs.TRANSFER_LEARNING:
            model=getattr(self.model_zoo, self.configs.MODEL)(weights='DEFAULT')
            
        else:
            model=getattr(self.model_zoo, self.configs.MODEL)()

        extra_layer=nn.Linear(self.check_output_dim(model), self.configs.CLASSES)
        model=AddLayer(model,extra_layer)

        self.checkpoint_path=os.path.join(self.configs.CHECKPOINT_PATH, self.configs.MODEL,'best.pt')
        os.makedirs(os.path.dirname(self.checkpoint_path),exist_ok=True)
        model=self.checkpoint_loader(model)
        model=self.on_cuda(model)

        return model

    def check_output_dim(self, model):
        dummy=torch.zeros((1, 3, self.configs.IMAGE_SIZE, self.configs.IMAGE_SIZE))
        output_dim=list(model(dummy).shape)
        return output_dim[1]

    def on_cuda(self, model):
        return ToCUDA(model)
        
    def checkpoint_loader(self, model):
        if self.configs.LOAD_CHECKPOINT and os.path.isfile(self.checkpoint_path):
            print("loading checkpoint ...")
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(self.checkpoint_path))
            else:
                model.load_state_dict(torch.load(self.checkpoint_path,map_location=torch.device('cpu')))
        return model

    def change_state(self, state, model):
        if state=="train":
            print("change model state to train !")
            model.train()
        else:
            print("change model state to eval !")
            model.eval()
        return model

    def DefineDataLoaders(self):
        print("Defining Dataloaders ...")
        train_dataset=CustomDataSet(self.configs.TRAIN_PATH,self.configs.IMAGE_SIZE)
        print(len(train_dataset) , "images found for training ...")
        train_loader=torch.utils.data.DataLoader(dataset=train_dataset,num_workers= self.configs.NUM_OF_WORKERS,
                                            batch_size= self.configs.BATCH_SIZE,sampler=train_dataset.balancer())

        val_dataset=CustomDataSet(self.configs.VALIDATION_PATH,self.configs.IMAGE_SIZE)
        print(len(val_dataset) , "images found for validation ...")
        val_loader=torch.utils.data.DataLoader(dataset=val_dataset ,num_workers= self.configs.NUM_OF_WORKERS
                                                ,batch_size= self.configs.BATCH_SIZE)
        return train_loader , val_loader

    def validation_loop(self,inp,Loss_fn):
        with torch.no_grad():
            L=0
            total=0
            correct=0
            for Images,targs in inp:
                Images=self.on_cuda(Images)
                targs=self.on_cuda(targs)

                outputs=self.model(Images)
                targs=targs.squeeze(1)
                L += Loss_fn(outputs,targs).detach().item()
                outputs=self.softmax(outputs)
                _,preds=torch.max(outputs.data,1)
                total += targs.size(0)
                if torch.cuda.is_available():
                    correct += (preds.cpu()==targs.cpu()).sum()
                else:
                    correct += (preds==targs).sum()
            Acc=correct/total
            L=L/len(inp)
            return Acc , L

    def TrainLoop(self):
        self.model=self.change_state("train", self.define_and_load_model())
        train_loader , val_loader=self.DefineDataLoaders()
        print("initialize optimizer ... [set to Adam @ lr={}]".format(self.configs.LR))
        Optimizer=torch.optim.Adam(self.model.parameters(),lr=self.configs.LR)
        # scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(Optimizer,T_max= self.configs.T_MAX_COEFFICIENT
        #                                                         *len(train_loader),eta_min=5e-14)
        scheduler=torch.optim.lr_scheduler.CyclicLR(Optimizer, base_lr=self.configs.BASE_LR, max_lr=self.configs.LR,
                                                    step_size_up=self.configs.T_MAX_COEFFICIENT*len(train_loader),
                                                    mode="triangular2", cycle_momentum=False)
        Loss_fn=torch.nn.CrossEntropyLoss(label_smoothing= self.configs.LABEL_SMOOTHING)
        print("Start training process ....")
        itert=0
        Flag=True
        optimizer_flag=True
        TLoss=0
        for e in range( self.configs.MAX_EPOCHS):
            for Images,targs in train_loader:
                itert += 1
                Images=self.on_cuda(Images)
                targs=self.on_cuda(targs)

                if Flag:
                    before=clone_weights(self.model)

                outputs=self.model(Images)
                loss=Loss_fn(outputs,targs.squeeze(1))
                loss.backward()
                Optimizer.step()

                if Flag:
                    Flag=False
                    after=clone_weights(self.model)
                    print(weights_checker(before,after))

                Optimizer.zero_grad()
                
                if itert % 10==0:
                    # Flag=True
                    val_acc, val_L=self.validation_loop(val_loader, Loss_fn)
                    train_acc, train_L=self.validation_loop(train_loader, Loss_fn)

                    self.writer.add_scalar('Loss/train', train_L, itert)
                    self.writer.add_scalar('Loss/validation', val_L, itert)
                    self.writer.add_scalar('Accuracy/train', train_acc, itert)
                    self.writer.add_scalar('Accuracy/validation', val_acc, itert)

                    print('after {} epochs and {} iterations'.format(e,itert) ,
                    'validation accuracy : {} % , validation Loss : {} train accuracy : {} % , train Loss : {}'.format(val_acc,val_L,train_acc,train_L))
                    if val_acc >= self.DesiredAccuracy:
                        self.DesiredAccuracy=val_acc 
                        torch.save(self.model.state_dict(),os.path.join(self.configs.CHECKPOINT_PATH
                                                                        , self.configs.MODEL,'best.pt'))
                        with open(os.path.join(self.configs.CHECKPOINT_PATH
                                            , self.configs.MODEL,'best.txt') , "w") as f:
                            f.write('accuracy is {} % , Loss is {}'.format(val_acc,val_L))
                            
                    self.writer.add_scalar('Learning rate', scheduler.get_last_lr()[0], itert)
                    print("current lr is : ",scheduler.get_last_lr())            
                scheduler.step()
            if e >=  self.configs.THRESHOLD_EPOCH and optimizer_flag:
                optimizer_flag=False
                Flag=True
                print("changing optimizer 2 SGD @ {} & CALR scheduler ...".format( self.configs.LR))
                Optimizer=torch.optim.SGD(self.model.parameters(),lr= self.configs.LR)
                # scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(Optimizer,T_max= self.configs.T_MAX_COEFFICIENT*len(train_loader),eta_min=5e-14)
                scheduler=torch.optim.lr_scheduler.CyclicLR(Optimizer, base_lr=self.configs.BASE_LR, max_lr=self.configs.LR,
                                                    step_size_up=self.configs.T_MAX_COEFFICIENT*5*len(train_loader),mode="triangular2")
                # scheduler=torch.optim.lr_scheduler.ExponentialLR(Optimizer, gamma=0.95)

    def CheckModel(self):
        ''' torcheck '''
        # torcheck.register(Optimizer)
        # torcheck.add_module_changing_check(self.model, module_name="my_model")
        # torcheck.add_module_nan_check(self.model)
        # torcheck.add_module_inf_check(self.model)
        pass

    def get_activation(self,name):
        def hook(model, input, output):
            self.activation[name]=output.detach()
        return hook

    def RegisterHook(self):
        self.model.model.features.register_forward_hook(self.get_activation('last_cnn'))
        self.model.model.avgpool.register_forward_hook(self.get_activation('last_pool'))

    @staticmethod
    def percision(array , idx , label):
        p=array[idx,idx]/(array.sum(1)[idx])
        # print(f"percision for class {label} : {p}")
        return p

    @staticmethod
    def recall(array , idx , label):
        r=array[idx,idx]/(array.sum(0)[idx])
        # print(f"recall for class {label} : {r}")
        return r
        
    @staticmethod
    def f1_score(p,r):
        return 2*p*r/(p+r)

    def result(self, inp, lbl=None):
        self.inp_path=inp
        self.lbl=lbl
        self.cam_extractor={}
        for cam_method in self.camapp:
            self.cam_extractor[cam_method]=self.camapp[cam_method](self.models[cam_method])
        self.iou_sum=0
        self.count=0

        if os.path.isfile(inp):
            self.inference(inp, lbl, self.configs.MODE)
        else:
            if self.configs.MODE=="LRP":
                results_path=os.path.join(self.configs.CAM_PATH, \
                                          self.configs.MODEL, "_".join(self.xai_mode))
            elif self.configs.MODE=="CAM":
                results_path=os.path.join(self.configs.CAM_PATH, \
                                          self.configs.MODEL, "_".join(self.xai_mode))
            if os.path.isdir(results_path):
                shutil.rmtree(results_path)
            subdirs=sorted(os.listdir(inp))

            confusion_mat_path=os.path.join(results_path, "confusion_matrix")
            os.makedirs(confusion_mat_path, exist_ok=True)

            if len(subdirs) != self.configs.CLASSES:
                raise Exception("wrong input path !")

            confusion_array=np.zeros((self.configs.CLASSES, self.configs.CLASSES))
            for label,subdir in enumerate(subdirs):
                imgs=glob.glob(os.path.join(inp,subdir,"*"))
                for img in tqdm(imgs):
                    pred, score=self.inference(img, label, self.configs.MODE, results_path)
                    confusion_array[pred, label]+=1

            with open(os.path.join(confusion_mat_path,"results.txt"), "w") as f:
                for idx in range(self.configs.CLASSES):
                    p=main.percision(confusion_array, idx, subdirs[idx])
                    r=main.recall(confusion_array, idx, subdirs[idx])
                    f1=main.f1_score(p,r)
                
                    f.write(f"Precision for {subdirs[idx]} : {round(p,4)}"+"\n")
                    f.write(f"Recall for {subdirs[idx]} : {round(r,4)}"+"\n")
                    f.write(f"F1 score for {subdirs[idx]} : {round(f1,4)}"+"\n"+"\n") 

            with open(os.path.join(confusion_mat_path,"IOU.txt"), "w") as f:
                f.write(f"Average IOU for {self.configs.CAM} : {round((self.iou_sum/self.count),4)}"+"\n")
                f.write(f"Total number of masks : {self.count}")


            df_cm=pd.DataFrame(confusion_array, index=[i for i in subdirs],
                  columns=[i for i in subdirs])
            plt.figure(figsize=(10,10))
            sn.heatmap(df_cm, annot=True)
            plt.savefig(os.path.join(confusion_mat_path,"confusion.png"))

    def IOU(self, heatmap, mask):
      h, w=heatmap.shape
      mask=cv2.imread(mask)
      if len(mask.shape) != 2:
        #   print(f"{mask.shape[-1]} channel image, Converting to 1 channel gray ...")
          mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
      mask=cv2.resize(mask, (w, h))

      intersection=np.logical_and((mask==255), (heatmap==255)).sum()
      union=np.logical_or((mask==255), (heatmap==255)).sum()
      return intersection/union

    def voter(self, inp):
        h, w=inp[0].shape
        ensembled=np.zeros((h,w)).astype("uint8")
        for i in inp:
            ensembled=np.logical_or((ensembled==255),(i==255)).astype("uint8") * 255
        return ensembled

    def ensemble_cam(self, img):
        
        activations=[]
        for cam_method in self.camapp:
            out=self.models[cam_method](img)
            out=self.softmax(out)
            score,pred=torch.max(out.data , 1)
            pred=int(pred)
            score=round(score.item(),2)
            activation_map=self.cam_extractor[cam_method](pred, out)
            im=activation_map[0].squeeze(0).detach().cpu().numpy()
            im=(im - im.min()) / (im.max() - im.min()) #to normalize
            im=(im * 255).astype(np.uint8)
            im[im>self.configs.HEATMAP_THRESH]=255
            im[im<=self.configs.HEATMAP_THRESH]=0
            h, w=im.shape
            activations += [im]

        return self.voter(activations), pred, score
    
    def lrp(self, inp):
        activations=[]
        for rule in self.configs.LRP_RULE:
            img=inp.detach().clone()
            lrp_vgg = self.on_cuda(lrp.convert_vgg(self.model))
            img.requires_grad_(True)
            out=self.model(img)
            out=self.softmax(out)
            score,pred=torch.max(out.data , 1)
            pred=int(pred)
            score=round(score.item(),2)
            out_lrp = lrp_vgg.forward(img, explain=True, rule=rule)
            out_lrp = out_lrp[torch.arange(1), out_lrp.max(1)[1]] # Choose maximizing output neuron
            out_lrp = out_lrp.sum()
            # Backward pass (do explanation)
            out_lrp.backward()
            explanation=img.grad.squeeze(0)
            explanation=explanation.detach().cpu().numpy()
            explanation=(explanation - np.min(explanation))/(np.max(explanation) - np.min(explanation))
            explanation=np.transpose(explanation,(1,2,0))
            explanation*=255
            explanation=explanation.astype("uint8")
            _, _, c=explanation.shape
            if c==3:
                image = Image.fromarray(explanation).convert('L')
            activations+=[np.array(image)]

        return self.voter(activations), pred, score
    
    def middle_layer(self, img):
        img_name=os.path.splitext(os.path.basename(img))[0]
        img=Image.open(img)
        img=img.resize(( self.configs.IMAGE_SIZE, self.configs.IMAGE_SIZE))
        if len(np.array(img).shape)==2:
            img=Image.fromarray(np.stack((np.array(img),)*3, axis=-1))
        src=img.copy()
        w,h=img.size
        if torch.cuda.is_available():
            img=PrepareData(img).unsqueeze_(0).cuda()
        else:
            img=PrepareData(img).unsqueeze_(0)
        self.model(img)
        return self.activation["last_pool"]

    def inference(self, img, label, mode, save_path=None):
        #lrp=LRP(self.model)
        if not save_path:
            os.makedirs(os.path.join(self.configs.CAM_PATH, self.configs.MODEL), exist_ok=True)
            subdirs=sorted(os.listdir(self.configs.TRAIN_PATH))
        else:
            subdirs=sorted(os.listdir(self.inp_path))
        img_name=os.path.splitext(os.path.basename(img))[0]
        img=Image.open(img)
        img=img.resize(( self.configs.IMAGE_SIZE, self.configs.IMAGE_SIZE))
        if len(np.array(img).shape)==2:
            img=Image.fromarray(np.stack((np.array(img),)*3, axis=-1))
        src=img.copy()
        w,h=img.size
        if torch.cuda.is_available():
            img=PrepareData(img).unsqueeze_(0).cuda()
        else:
            img=PrepareData(img).unsqueeze_(0)

        if mode=="CAM":
            im, pred, score=self.ensemble_cam(img)
        elif mode=="LRP":
            im, pred, score=self.lrp(img)

        if label is not None and not isinstance(label, str):
            if subdirs[label]==self.lbl:
                mask=glob.glob(os.path.join(self.configs.SEGMENTS_PATH,img_name+".*"))[0]
                self.iou_sum += self.IOU(im, mask)
                self.count += 1
        elif isinstance(label, str):
            mask=glob.glob(os.path.join(self.configs.SEGMENTS_PATH,img_name+".*"))[0]
            iou=self.IOU(im, mask)
            print(f"IOU is : {iou}")
        result=overlay_mask(src, Image.fromarray(im/255), alpha=0.5)
        if not save_path:
            single_image_path=os.path.join(self.configs.CAM_PATH, self.configs.MODEL, "single_image")
            os.makedirs(single_image_path, exist_ok=True)
            result.save(os.path.join(single_image_path,f"{img_name}_{label}_{subdirs[pred]}_{'_'.join(self.xai_mode)}.png"))
            print(f"prediction : {subdirs[pred]} - confidence : {score}")
        else:
            result.save(os.path.join(save_path,f"{img_name}_{subdirs[label]}_{subdirs[pred]}_{'_'.join(self.xai_mode)}.png"))
            return pred , score

