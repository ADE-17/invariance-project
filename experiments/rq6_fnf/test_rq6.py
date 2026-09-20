"""Unit checks for RQ6 components; run on CPU without cluster data."""
import unittest
import numpy as np
import pandas as pd
import torch
from experiments.rq6_fnf import fnf
from experiments.rq6_fnf.data import apply_condition


class FlowTests(unittest.TestCase):
    def test_encode_decode_inverse_and_logdet(self):
        torch.manual_seed(0)
        d = 4
        flow = fnf.Flow(d, [16, 16], 4, fnf.alternating_masks(d, 4, np.random.default_rng(0)))
        x = torch.randn(8, d)
        z, ld = flow.encode(x)
        xr, ld_back = flow.decode(z)
        self.assertTrue(torch.allclose(x, xr, atol=1e-5))
        self.assertTrue(torch.allclose(ld, -ld_back, atol=1e-5))
        jac = torch.autograd.functional.jacobian(lambda v: flow.encode(v[None])[0][0], x[0])
        self.assertAlmostEqual(float(torch.logdet(jac).abs()), abs(float(ld[0])), places=4)

    def test_flow_prior_log_prob_finite_and_sampling(self):
        torch.manual_seed(0)
        d = 3
        prior = fnf.FlowPrior(d, [8], 2, fnf.alternating_masks(d, 2, np.random.default_rng(1)))
        x = prior.sample(16)
        self.assertEqual(x.shape, (16, d))
        self.assertTrue(torch.isfinite(prior.log_prob(x)).all())

    def test_gmm_prior_and_symmetric_kl(self):
        rng = np.random.default_rng(0)
        x0, x1 = rng.normal(0, 1, (400, 3)).astype('float32'), rng.normal(1, 1, (400, 3)).astype('float32')
        _, p0, _ = fnf.fit_gmm_prior(x0, x0, 2, 0, 'cpu')
        _, p1, _ = fnf.fit_gmm_prior(x1, x1, 2, 0, 'cpu')
        masks = fnf.alternating_masks(3, 2, np.random.default_rng(0))
        flows = torch.nn.ModuleList([fnf.Flow(3, [8], 2, masks) for _ in range(2)])
        kl, hits = fnf.symmetric_kl(flows, [p0, p1], 256)
        self.assertTrue(torch.isfinite(kl))
        dist, _ = fnf.statistical_distance(flows, [p0, p1], n=500)
        self.assertTrue(0 <= dist <= 1)

    def test_symmetric_kl_matches_gaussian_shift_and_penalises_contraction(self):
        # Two unit Gaussians shifted by 1 in every coordinate: symmetric KL per direction is d/2, so the
        # averaged estimator must be close to d/2 for near-identity flows and must not drop when a flow
        # contracts volume (the log|det| sign error rewarded contraction and drove the KL to -55 nats).
        rng = np.random.default_rng(0); torch.manual_seed(0); d = 4
        x0 = rng.normal(0, 1, (20000, d)).astype('float32'); x1 = rng.normal(1, 1, (20000, d)).astype('float32')
        _, p0, _ = fnf.fit_gmm_prior(x0, x0, 1, 0, 'cpu'); _, p1, _ = fnf.fit_gmm_prior(x1, x1, 1, 0, 'cpu')
        masks = fnf.alternating_masks(d, 4, np.random.default_rng(0))
        flows = torch.nn.ModuleList([fnf.Flow(d, [16], 4, masks) for _ in range(2)])
        with torch.no_grad():
            for f in flows:
                for prm in f.parameters(): prm.mul_(0.01)
            base, _ = fnf.symmetric_kl(flows, [p0, p1], 20000)
            for layer in flows[0].layers:
                layer.s[-2].bias.fill_(2.0)
            contracted, _ = fnf.symmetric_kl(flows, [p0, p1], 20000)
            ld = flows[0].encode(p0.sample(2000))[1].mean()
        self.assertAlmostEqual(float(base), d / 2, delta=0.3)
        self.assertLess(float(ld), -3)
        self.assertGreater(float(contracted), float(base))

    def test_dequantizer_range(self):
        x = np.array([[0, 1, 5], [2, 1, 7], [4, 3, 9]], dtype='float32')
        dq = fnf.Dequantizer(.05).fit(x)
        s = dq.scale(np.array([[3, 2, 100]], dtype='float32'))
        self.assertTrue(((s > 0) & (s < 1)).all())
        out = fnf.dequantize(torch.from_numpy(s), torch.tensor(dq.q, dtype=torch.float32), .05, torch.Generator().manual_seed(0))
        self.assertTrue(torch.isfinite(out).all())


class CategoricalTests(unittest.TestCase):
    def test_index_roundtrip_and_one_hot(self):
        dims = [3, 2, 4]
        codes = np.array([[0, 0, 0], [2, 1, 3], [1, 1, 2]])
        idx = fnf.combination_index(codes, dims)
        self.assertTrue(np.array_equal(fnf.index_to_codes(idx, dims), codes))
        oh = fnf.one_hot(codes, dims)
        self.assertEqual(oh.shape, (3, 9))
        self.assertTrue((oh.sum(1) == 3).all())

    def test_rank_matching_is_bijective_and_zero_distance_when_equal(self):
        rng = np.random.default_rng(0)
        logp = np.log(rng.dirichlet(np.ones(24)))
        cls = rng.integers(0, 2, 24)
        pg, pa, dist = fnf.rank_matching_maps(logp, logp, cls, 1.)
        self.assertEqual(len(set(pg.tolist())), 24)
        self.assertEqual(len(set(pa.tolist())), 24)
        self.assertAlmostEqual(dist, 0., places=10)
        other = np.log(rng.dirichlet(np.ones(24)))
        _, _, dist2 = fnf.rank_matching_maps(logp, other, cls, 0.)
        self.assertGreater(dist2, 0.)

    def test_encode_categorical_keeps_reference_group(self):
        dims = [2, 3]
        codes = np.array([[0, 0], [1, 2], [1, 1], [0, 2]])
        a = np.array([1, 0, 1, 0])
        total = 6
        pg = np.roll(np.arange(total), 1)
        pa = np.roll(np.arange(total), 2)
        z, idx = fnf.encode_categorical(codes, a, dims, pg, pa, .5, 1, 0)
        self.assertTrue(np.array_equal(idx[a == 1], fnf.combination_index(codes[a == 1], dims)))
        self.assertTrue((z.sum(1) == 2).all())

    def test_made_is_autoregressive_and_normalised_over_binary_vectors(self):
        torch.manual_seed(0)
        d = 5
        made = fnf.MADE(d, [12, 12], np.random.default_rng(3))
        x = torch.rand(1, d)
        jac = torch.autograd.functional.jacobian(lambda v: made(v[None])[0], x[0])
        degrees = np.arange(d)
        for i in range(d):
            for j in range(d):
                if not degrees[j] < degrees[i]:
                    self.assertEqual(float(jac[i, j].abs()), 0., f'output {i} depends on input {j}')
        grid = torch.tensor([[(k >> b) & 1 for b in range(d)] for k in range(2 ** d)], dtype=torch.float32)
        self.assertAlmostEqual(float(made.log_prob(grid).exp().sum()), 1., places=5)

    def test_made_log_prob_normalises_over_combinations(self):
        torch.manual_seed(0)
        dims = [2, 3]
        made = fnf.MADE(5, [8, 8], np.random.default_rng(0))
        logp = fnf.enumerate_log_probs(made, dims, 'cpu')
        self.assertAlmostEqual(float(np.exp(logp).sum()), 1., places=5)


class ConditionTests(unittest.TestCase):
    def frame(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2, 4000)
        y = (rng.random(4000) < np.where(a == 0, .2, .4)).astype(int)
        return pd.DataFrame(dict(a=a, y=y, unit=np.arange(4000).astype(str)))

    def test_shift_halves_group0_positives(self):
        d = self.frame()
        s = apply_condition(d, 'shift50', 'train')
        before, after = ((d.a == 0) & (d.y == 1)).sum(), ((s.a == 0) & (s.y == 1)).sum()
        self.assertAlmostEqual(after / before, .5, delta=.01)
        self.assertEqual(((s.a == 1) & (s.y == 1)).sum(), ((d.a == 1) & (d.y == 1)).sum())

    def test_equalized_matches_rates(self):
        s = apply_condition(self.frame(), 'equalized', 'test')
        rates = s.groupby('a').y.mean()
        self.assertAlmostEqual(rates[0], rates[1], delta=.005)

    def test_natural_unchanged(self):
        d = self.frame()
        self.assertTrue(apply_condition(d, 'natural', 'train').equals(d))


class TestCalibration(unittest.TestCase):

    def test_cox_calibration_recovers_known_slope_and_intercept(self):
        from scipy.special import expit
        from experiments.rq6_fnf.calibration import cox_calibration, group_calibration
        rng = np.random.default_rng(0)
        z = rng.normal(size=200_000)
        y = rng.binomial(1, expit(z))
        good = cox_calibration(y, expit(z))
        assert abs(good['slope'] - 1) < .02 and abs(good['intercept']) < .02 and abs(good['intercept_fixed_slope']) < .02
        under = cox_calibration(y, expit(.5 * z))
        assert abs(under['slope'] - 2) < .05
        over = cox_calibration(y, expit(2 * z + 1))
        assert over['slope'] < 1 and over['intercept'] < 0 and over['slope_ci_low'] < over['slope'] < over['slope_ci_high']
        # a > 0 when predictions are systematically too low (observed exceeds predicted)
        low = cox_calibration(y, expit(z - 1))
        assert .95 < low['intercept_fixed_slope'] < 1.05 and low['mean_bias'] > 0
        degenerate = cox_calibration(np.ones(10), np.full(10, .5))
        assert np.isnan(degenerate['slope'])
        a = rng.integers(0, 2, size=y.size)
        shifted = expit(z - a)  # group 1 under-predicted by one logit
        g = group_calibration(y, shifted, a)
        assert abs(g['group0_cal_intercept_fixed_slope']) < .05 and .95 < g['group1_cal_intercept_fixed_slope'] < 1.05
        assert .9 < g['cal_intercept_fixed_slope_gap'] < 1.1


if __name__ == '__main__':
    unittest.main()
