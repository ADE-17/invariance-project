import unittest, numpy as np, torch
from experiments.rq8_final.methods import *

class NumericalTests(unittest.TestCase):
    def test_flow_roundtrip_and_density(self):
        torch.manual_seed(42)
        f=Flow(4,[8],4,alternating_masks(4,4,np.random.default_rng(42)))
        x=torch.randn(40,4);z,ld=f.encode(x);xx,ild=f.decode(z)
        torch.testing.assert_close(x,xx,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(ld,-ild,atol=1e-5,rtol=1e-5)
        prior=FlowPrior(4,[8],2,alternating_masks(4,2,np.random.default_rng(5)))
        direct=prior.log_prob(x)-ld;indirect=latent_log_density(z,prior,f)
        torch.testing.assert_close(direct,indirect,atol=2e-5,rtol=2e-5)
        loss=symmetric_kl([f,f],[prior,prior],200)
        self.assertLess(abs(float(loss)),1e-5)
    def test_affine_kl_analytic(self):
        class Scale:
            def __init__(self,s):self.s=s
            def encode(self,x):return x*self.s,torch.full((len(x),),np.log(self.s))
            def decode(self,z):return z/self.s,torch.full((len(z),),-np.log(self.s))
        class Normal:
            def sample(self,n):return torch.randn(n,1)
            def log_prob(self,x):return torch.distributions.Normal(0,1).log_prob(x).sum(1)
        torch.manual_seed(1)
        # Half the two Gaussian KLs for N(0,1) and N(0,4) = 0.5625.
        kl=symmetric_kl([Scale(1.),Scale(2.)],[Normal(),Normal()],100000)
        self.assertAlmostEqual(float(kl),.5625,delta=.025)
    def test_discrepancy_detects_and_backpropagates(self):
        torch.manual_seed(1);a=torch.arange(512)%2
        x=(torch.randn(512,8)+a[:,None]*3).requires_grad_()
        for fn in [mmd,hsic]:
            l=fn(x,a);self.assertGreater(float(l),.05);l.backward(retain_graph=True)
        self.assertTrue(torch.isfinite(x.grad).all())
        z=torch.cat([x[:100].detach(),x[:100].detach()]);aa=torch.cat([torch.zeros(100),torch.ones(100)]).long()
        self.assertLess(abs(float(mmd(z,aa))),1e-5)
    def test_score_channel(self):
        rng=np.random.default_rng(42);a=rng.integers(0,2,2000);p=np.clip(rng.random(2000)*.5+a*.4,.001,.999);y=(rng.random(2000)<p).astype(int)
        for aware in [False,True]:
            m=ScoreRelease(16,aware).fit(p,y,a)
            self.assertLessEqual(m.train_tv,.010001)
            np.testing.assert_allclose(m.K.sum(1),1)
            out=m.transform(p,a,42,10);self.assertEqual(out.shape,(2000,10))
            np.testing.assert_array_equal(out,m.transform(p,a,42,10))
    def test_grid(self):
        g=grid();self.assertEqual(len(g),10);self.assertEqual(len(set(x['id'] for x in g)),len(g))
    def test_weighted_ap_ties(self):
        from experiments.rq8_final.evaluate import weighted_ap
        from sklearn.metrics import average_precision_score
        y=np.array([0,1,1,0,1]);p=np.array([.5,.5,.8,.1,.1]);w=np.array([[1,2,1,3,2],[1,1,1,1,1.]])
        got=weighted_ap(y,p,w)
        for i in range(2):self.assertAlmostEqual(got[i],average_precision_score(y,p,sample_weight=w[i]))
    def test_shift_exact_and_reproducible(self):
        import pandas as pd
        from experiments.rq8_final.prepare import shift50
        d=pd.DataFrame(dict(a=[0]*11+[0]*5+[1]*12,y=[1]*11+[0]*5+[1]*12))
        x,removed=shift50(d,19);xx,again=shift50(d,19)
        self.assertEqual(len(removed),5);self.assertEqual(len(x),23)
        np.testing.assert_array_equal(removed,again)
        self.assertTrue(((d.loc[removed].a==0)&(d.loc[removed].y==1)).all())
    def test_deltas_and_weighted_metrics(self):
        from experiments.rq8_final.evaluate import extended,detailed_weighted,metric_deltas
        y=np.array([0,1,0,1,0,1,0,1]);a=np.array([0,0,0,0,1,1,1,1]);p=np.array([.1,.7,.8,.9,.2,.3,.6,.8])
        point=extended(y,p,a,.5);weighted=detailed_weighted(y,p,a,.5,np.ones((1,len(y))))
        for k,v in weighted.items():self.assertAlmostEqual(v[0],point[k],msg=k)
        self.assertTrue(all(v==0 for v in metric_deltas(point,point).values()))

if __name__=='__main__':unittest.main()
