import unittest
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

import symbtools as st
import pyblocksim as pbs

import boasim as bs

from ipydex import IPS, Container, activate_ips_on_exception


activate_ips_on_exception()

# noinspection PyPep8Naming
class TestCore(unittest.TestCase):
    def setUp(self):
        pbs.restart()

    def test_010__block_10a__DirectionSensitiveSigmoid(self):

        T = bs.T
        k = bs.k
        T_trans_pos = 3
        T_trans_neg = 5
        u_amplitude = 20

        step1 = 10
        step2 = T_trans_pos/T + step1 + 25

        u1_expr = pbs.sp.Piecewise((0, k < step1), (u_amplitude, k < step2), (0, True))

        dss_1 = bs.dtDirectionSensitiveSigmoid(
            input1=u1_expr,
            params=dict(K=1, T_trans_pos=T_trans_pos, T_trans_neg=T_trans_neg, sens=.1, f_wait_neg=0.3)
        )

        N_steps = int(step2 + T_trans_neg/T)+10
        kk, xx, bo = pbs.td.blocksimulation(N_steps)

        steps_start = np.r_[step1*T, step2*T]
        steps_end = steps_start + np.r_[T_trans_pos, T_trans_neg]

        if 0:
            plt.plot(kk*T, dss_1.output_res, marker=".")
            plt.plot(kk*T, xx[:, 4], marker=".")
            plt.vlines(steps_start, ymin=-1, ymax=u_amplitude, colors="tab:pink")
            plt.vlines(steps_end, ymin=-1, ymax=u_amplitude, colors="k")
            plt.grid()
            plt.show()

        # now simulate again but with sympy_to_c
        kk2, xx2, bo = pbs.td.blocksimulation(N_steps, rhs_options={"use_sp2c": True})

        # compare lambdify-result and c-result
        self.assertTrue(np.allclose(xx - xx2, 0))

    def test_020__block_10b__Sufenta(self):

        dc = Container()

        dc.cr1 = dc.cr2 = 0.3
        T = bs.T
        t = bs.t

        T_end = 120
        tt = np.arange(0, int(T_end/T) + 1)*T

        u_expr_sufenta = sp.Piecewise((dc.cr1, apx(t, 5)), (dc.cr2, apx(t, 40)), (0, True))
        u_func = st.expr_to_func(t, u_expr_sufenta)

        params = dict(
            rise_time = 5,
            down_slope = -.8/15,
            active_time_coeff = 100,
            dose_gain = 0.5/0.3  # achieve output of 0.5 for 0.3 mg/kgKG
        )

        sufenta_block = bs.dtSufenta(input1=u_expr_sufenta, params=params)

        N_steps = int(90/T)
        kk, xx, bo = pbs.td.blocksimulation(N_steps)

        # now simulate again but with sympy_to_c
        # `cleanup = False` helps to debug the c-code generation
        kk2, xx2, bo = pbs.td.blocksimulation(N_steps, rhs_options={"use_sp2c": True, "sp2c_cleanup": False})

        # compare lambdify-result and c-result
        self.assertTrue(np.allclose(xx - xx2, 0))

    def test_030__block_10c__Akrinor(self):

        T1 = 5 # dc.t_cr1
        T2 = 22.5 # dc.t_cr2

        # see notebook 07c_akrinor for where this comes from:
        acrinor_block_dose_gain = 5.530973451327434
        T = bs.T
        t = bs.t

        body_mass = 70

        relative_dose_akri = 0.02 # dc.cr1

        dose_akri = relative_dose_akri * body_mass

        u_expr_acrinor = sp.Piecewise((dose_akri, apx(t, T1)), (dose_akri, apx(t, T2)), (0, True))

        l1 = pbs.td.get_loop_symbol()
        bp_sum  = pbs.td.StaticBlock(output_expr=60 + l1)
        bp_delay_block = pbs.td.dtDelay1(input1=bp_sum.Y)

        T_end = 90

        params = dict(
            T_75=5,  # min
            T_plateau = 30,  # min (including rising phase)
            down_slope = -1,  # mmHg/min
            body_mass = 70,
            dose_gain = acrinor_block_dose_gain, # [1/(ml/kgKG)]
        )

        akrinor_block = bs.dtAkrinor(input1=u_expr_acrinor, input2=bp_delay_block.Y, params=params)
        pbs.td.set_loop_symbol(l1, akrinor_block.Y)

        N_steps = int(T_end/T)

        # activate_ips_on_exception()

        # test for a bug that rhs returns different expressions each time
        test_expr1 = akrinor_block.rhs(0, (0,)*11)[2].args[0].args[1]
        test_expr2 = akrinor_block.rhs(0, (0,)*11)[2].args[0].args[1]

        self.assertEqual(test_expr1, test_expr2)

        # for the numeric comparison simulate with sympy_to_c and with lambdify
        kk, xx, bo = pbs.td.blocksimulation(N_steps)
        kk2, xx2, bo = pbs.td.blocksimulation(N_steps, rhs_options={"use_sp2c": True})

        if 0:
            plt.plot(kk*T, bo[bp_sum], label="Akrinor effect")
            plt.show()

        # output signal:

        y = bo[bp_sum]

        y_expected = np.array([66.5, 73.8])
        self.assertTrue(np.allclose(y[[199, 349]], y_expected, atol=.1))

        # compare lambdify-result and c-result
        self.assertTrue(np.allclose(xx - xx2, 0))

    def test_040__block_10d__Propofol_bolus(self):

        # for debugging warnings:
        # import warnings
        # warnings.filterwarnings("error")

        t = bs.t
        T = bs.T
        T_end = 10
        bp_normal = 100
        bis_normal = 1

        u_expr_propofol_boli = sp.Piecewise((0.5, apx(t, 0)), (1.5, apx(t, 2)), (0, True))
        N_steps = int(T_end/T)

        l1 = pbs.td.get_loop_symbol()
        bp_sum  = pbs.td.StaticBlock(output_expr=100 + l1)
        bp_delay_block = pbs.td.dtDelay1(input1=bp_sum.Y)


        params = dict(reference_bp=bp_normal, reference_bis=bis_normal)

        pfl = bs.dtPropofolBolus(input1=u_expr_propofol_boli, input2=bp_delay_block.Y, params=params)
        pbs.td.set_loop_symbol(l1, pfl.Y[0])

        # initial values
        iv = {pfl.x1: 100, pfl.x2: 1}
        kk, xx, bo = pbs.td.blocksimulation(N_steps, iv=iv)

        # again with sympy_to_c
        kk, xx2, bo = pbs.td.blocksimulation(N_steps, iv=iv, rhs_options={"use_sp2c": True})

    def test_050__2D_output(self):

        t_discrete = pbs.td.k*pbs.td.T

        class TwoDOutputBlock(pbs.td.StaticBlock):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # create a column vector (one-column matrix)
                self.output_expr_2d = sp.Matrix([self.output_expr, self.output_expr*2])

            def output(self):
                return self.output_expr_2d

        static_block_2d  = TwoDOutputBlock(output_expr=sp.sin(sp.pi*t_discrete))
        bp_delay_block = pbs.td.dtDelay1(input1=static_block_2d.Y[1])

        # initial values
        iv = {}
        N_steps = 30
        kk, xx, bo = pbs.td.blocksimulation(N_steps, iv=iv)
        self.assertEqual(bo[static_block_2d].shape, (N_steps, 2))
        self.assertTrue(np.allclose(bo[static_block_2d][0, :], [0, 0]))
        self.assertTrue(np.allclose(bo[static_block_2d][5, :], [1, 2]))


# #################################################################################################
#
# auxiliary functions
#
# #################################################################################################


def apx(x, x0, eps=1e-3):
    """
    express condition that x \\approx x0
    """

    return (sp.Abs(x - x0) < eps)