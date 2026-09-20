import inspect
import unittest
import numpy as np
import torch
from scipy.special import expit,logit
from .methods import ConditionalFNF,conditional_kl,latent_log_density,mmd,grid
from .calibration import weak_calibration,diagnostics


class Tests(unittest.TestCase):
    def test_roundtrip_density(self):
        class Normal:
            def log_prob(self,x):return torch.distributions.Normal(0.,1.).log_prob(x).sum(1)
        torch.manual_seed(42);f=ConditionalFNF(4).flows[0];x=torch.randn(80,4)
        z,ld=f.encode(x);xx,ild=f.decode(z)
        torch.testing.assert_close(x,xx,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(ld,-ild,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(Normal().log_prob(x)-ld,latent_log_density(z,Normal(),f),atol=1e-5,rtol=1e-5)

    def test_conditional_not_marginal(self):
        # Same class distributions across groups despite different class prevalences.
        z=torch.tensor([[-1.]]*90+[[1.]]*10+[[-1.]]*10+[[1.]]*90)
        a=torch.tensor([0]*100+[1]*100);y=(z[:,0]>0).long()
        self.assertGreater(float(mmd(z,a)),.1)
        for label in (0,1):self.assertAlmostEqual(float(mmd(z[y==label],a[y==label])),0.,places=6)

    def test_conditional_kl_analytic(self):
        class Scale:
            def __init__(self,s):self.s=s
            def encode(self,x):return x*self.s,torch.full((len(x),),np.log(self.s))
            def decode(self,x):return x/self.s,torch.full((len(x),),-np.log(self.s))
        class Normal:
            def sample(self,n):return torch.randn(n,1)
            def log_prob(self,x):return torch.distributions.Normal(0,1).log_prob(x).sum(1)
        torch.manual_seed(12)
        total,each=conditional_kl([Scale(1.),Scale(2.)],[[Normal(),Normal()],[Normal(),Normal()]],50000)
        self.assertAlmostEqual(float(total),.5625,delta=.025)
        self.assertTrue(all(abs(float(v)-.5625)<.035 for v in each))

    def test_no_label_inference_and_gradients(self):
        from .train import predict
        self.assertNotIn('y',inspect.signature(predict).parameters)
        self.assertEqual(list(inspect.signature(ConditionalFNF.forward).parameters),['self','x','a'])
        torch.manual_seed(1);model=ConditionalFNF(4);x=torch.randn(32,4);a=torch.arange(32)%2
        z,l=model(x,a);loss=l.square().mean()+z.square().mean();loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        model.eval()
        np.testing.assert_array_equal(predict(model,x.numpy(),a.numpy(),'cpu')[1],predict(model,x.numpy(),a.numpy(),'cpu')[1])

    def test_calibration_sign_constant_and_reference(self):
        from .calibration_reference import calibration_slope_intercept
        y=np.array([0]*80+[1]*20);p=np.full(100,.4)
        d=weak_calibration(y,p,np.repeat(np.arange(50),2))
        self.assertAlmostEqual(d['intercept_fixed_slope'],logit(.2)-logit(.4),places=6)
        self.assertGreater(d['mean_bias'],0);self.assertIsNone(d['slope'])
        rng=np.random.default_rng(4);z=rng.normal(size=3000);y=rng.binomial(1,expit(z));p=expit(2*z+1)
        got=weak_calibration(y,p);ref=calibration_slope_intercept(y,p)
        for k in ['intercept','slope','intercept_fixed_slope']:self.assertAlmostEqual(got[k],ref[k],places=7)
        self.assertLess(got['intercept'],0) # Supplied example's a>0 comment has the sign reversed.
        d=weak_calibration(np.ones(10),np.full(10,.2));self.assertIsNone(d['intercept_fixed_slope'])

    def test_grid_and_metric_integration(self):
        self.assertEqual([c['gamma'] for c in grid()],[.05,.5,.95])
        self.assertEqual(len(set(c['id'] for c in grid())),3)
        rng=np.random.default_rng(23);p=rng.uniform(.1,.9,500);y=rng.binomial(1,p);a=np.arange(500)%2
        metrics,detail=diagnostics(y,p,a,np.repeat(np.arange(250),2))
        self.assertIn('worst_abs_calibration_bias_score',metrics)
        self.assertEqual(detail['group0']['ci_method'],'patient_clustered_wald')
        self.assertTrue(detail['group0']['fixed_slope_converged'])


if __name__=='__main__':unittest.main()
