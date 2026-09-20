"""Fresh shifted ERM; train patients only, no missing-image placeholders."""
import sys, time, argparse, random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
from PIL import Image
from sklearn.metrics import roc_auc_score,log_loss
from experiments.rq8_final.common import *
from experiments.rq8_final.prepare import prepare

class Images(Dataset):
    def __init__(self,d):
        self.paths=d.image_path.tolist(); self.y=d.y.to_numpy().astype('float32')
        self.tf=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.y)
    def __getitem__(self,i):
        with Image.open(self.paths[i]) as image: x=self.tf(image.convert('RGB'))
        return x,self.y[i]

class Model(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        self.features=models.densenet121(weights=models.DenseNet121_Weights.DEFAULT if pretrained else None).features
        self.head=nn.Linear(1024,1)
    def forward(self,x):
        z=nn.functional.adaptive_avg_pool2d(nn.functional.relu(self.features(x)),1).flatten(1)
        return z,self.head(z).flatten()

@torch.no_grad()
def predict(model,dl,device):
    model.eval(); zs=[]; ps=[]
    for x,_ in dl:
        with torch.autocast('cuda',enabled=device=='cuda'):
            z,l=model(x.to(device,non_blocking=True))
        zs.append(z.float().cpu().numpy()); ps.append(l.float().sigmoid().cpu().numpy())
    return np.concatenate(zs),np.concatenate(ps)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true'); args=ap.parse_args()
    prepare(); sets=frames(); out=ROOT/'baseline';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists(): return
    torch.manual_seed(SEED); np.random.seed(SEED);random.seed(SEED)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    assert device=='cuda' or args.smoke
    if args.smoke: sets={s:d.iloc[:64].copy() for s,d in sets.items()}
    workers=0 if args.smoke else 4
    loaders={s:DataLoader(Images(d),batch_size=32,shuffle=s=='train',num_workers=workers,
        pin_memory=device=='cuda',persistent_workers=workers>0) for s,d in sets.items()}
    model=Model(not args.smoke).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
    scaler=torch.amp.GradScaler('cuda',enabled=device=='cuda')
    best=float('inf'); history=[]; start=time.time(); first=1; stale=0
    if (out/'last.pt').exists():
        ck=torch.load(out/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        history=ck['history'];best=ck['best'];stale=ck['stale'];first=ck['epoch']+1
        torch.set_rng_state(ck['rng_cpu'].cpu());
        if device=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in ck['rng_cuda']])
    write(out/'config.json',dict(pathology=PATHOLOGY,condition='shift50',seed=SEED,epochs=30,batch=32,lr=1e-4,architecture='DenseNet121',pretrained=not args.smoke,
       image_size=256,selection='minimum validation BCE; patience 7 after epoch 10',smoke=args.smoke))
    for epoch in range(first, (1 if args.smoke else 30)+1):
        begin=time.time();model.train();total=0
        for bi,(x,y) in enumerate(loaders['train']):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',enabled=device=='cuda'):
                _,l=model(x.to(device,non_blocking=True));loss=nn.functional.binary_cross_entropy_with_logits(l,y.to(device))
            scaler.scale(loss).backward();scaler.step(opt);scaler.update();total+=float(loss)*len(x)
            if bi==19 and epoch==first:log(f'BENCHMARK 20 batches seconds={time.time()-begin:.2f}; epoch batches={len(loaders["train"])}')
        _,p=predict(model,loaders['validation'],device)
        vl=log_loss(sets['validation'].y,p,labels=[0,1])
        row=dict(epoch=epoch,train_bce=total/len(sets['train']),validation_bce=vl,
            validation_auroc=roc_auc_score(sets['validation'].y,p),epoch_seconds=time.time()-begin,elapsed_seconds=time.time()-start)
        history.append(row);pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        if vl<best:
            best=vl;stale=0;torch.save(dict(model=model.state_dict(),epoch=epoch),out/'best.pt')
        else:stale+=1
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),epoch=epoch,
            history=history,best=best,stale=stale,rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all() if device=='cuda' else []),out/'last.pt')
        log(row)
        if epoch>=10 and stale>=7:break
    ck=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ck['model'])
    for s,d in sets.items():
        dl=DataLoader(Images(d),batch_size=64,num_workers=workers,pin_memory=device=='cuda')
        z,p=predict(model,dl,device)
        np.save(out/f'{s}_z.npy',z);np.save(out/f'{s}_p.npy',p)
        assert len(z)==len(d) and np.isfinite(z).all()
    write(out/'complete.json',dict(selected_epoch=ck['epoch'],epochs_run=history[-1]['epoch'],elapsed_seconds=time.time()-start,
        checkpoint_sha256=sha(out/'best.pt'),smoke=args.smoke))

if __name__=='__main__':main()
