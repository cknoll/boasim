

import numpy as np
import matplotlib.pyplot as plt
import pyblocksim as pbs
import symbtools as st
import sympy as sp


#Propofol: Herzfrequenz, Blutdruck, BIS

# originale Kurven, zu jeweils einem Array komprimiert

a = 0
b = 5
c = 9
d = 20

cp0 = 0
cp1 = 5
cp2 = 8
cp3 = 0
rr0 = 100 #Ausgangsblutdruck zum Zeitpunkt 0
fp = 6 #Volumenstatus und kardiale Kompensationsfähigkeit: 0 (gut) bis 10 (schlecht)
# neue Benennung: k2 realistisches Intervall [2, 10]


# verschobene Zeitpunkte
dt1 = 3
dt2 = 6

# originale Zeitpunkte
dt1 = 0
dt2 = 0

xp = np.array ([0,1,1.01,5+dt1,5.01+dt1,9+dt2,9.01+dt2,20+dt2])
yp = np.array([cp0,cp0,cp1,cp1,cp2,cp2,cp3,cp3])


xr1 = np.linspace(a, b, 50)
xr2 = np.linspace(b, c, 50)
xr3 = np.linspace(c, d, 50)

# original
# tt = np.concatenate((xr1, xr2, xr3))


tt = np.concatenate((xr1, xr2 + dt1, xr3 + dt2))


yr1 = (rr0-((cp1-cp0)*fp))+((cp1-cp0)*fp) / (1 + np.exp(4*xr1-10))
yr2 = (rr0-((cp1-cp0)*fp)-((cp2-cp1)*fp))+((cp2-cp1)*fp) / (1 + np.exp(4*xr2-26))
yr3 = (rr0-((cp1-cp0)*fp)-((cp2-cp1)*fp))+((rr0-(fp*cp3))-(rr0-(fp*cp2))) / (1 + np.exp(-2*xr3+28))


# blood pressure
yy_bp = np.concatenate((yr1, yr2, yr3))

# normal rest heart frequency

hf0 = 80

yr4 = ((rr0-((cp1-cp0)*fp))+((cp1-cp0)*fp) / (1 + np.exp(4*xr1-10)))-20
yr5 = ((rr0-((cp1-cp0)*fp)-((cp2-cp1)*fp))+((cp2-cp1)*fp) / (1 + np.exp(4*xr2-26)))-20
yr6 = ((rr0-((cp1-cp0)*fp)-((cp2-cp1)*fp))+((rr0-(fp*cp3))-(rr0-(fp*cp2))) / (1 + np.exp(-2*xr3+28)))-20

yy_hf = np.concatenate((yr4, yr5, yr6))



#BIS für Dosis 0, 5, 10, 0

x1 = np.linspace(a, b, 50)
x2 = np.linspace(b, c, 50)
x3 = np.linspace(c, d, 50)

y1 = 0.5+0.5 / (1 + np.exp(4*x1-10))
y2 = 0.2+0.3 / (1 + np.exp(4*x2-26))
y3 = 0.2+0.8 / (1 + np.exp(-2*x3+28))



# TODO: continue with other curves


def step_factory(y0, y1, x_step=0, tanh_scaling_factor=1e3):
    """
    Factory to create continuously approximated step functions.

    :param y0:       start value of the step function
    :param y1:       final value of the step function
    :param x_step:   x-value at which the step occurs
                     (better: middle point of the step).

    :param tanh_scaling_factor:
                     scaling of shifted x-value, such that the tanh-func
                     becomes more rectangular for bigger values. Default 1e3

    """
    # tanh maps R to (-1, 1)

    # first map R to (0, 1)
    # then map (0, 1) -> (y0, y1)

    dy = y1-y0

    def fnc(x, module=sp):
        """
        The step function (depends on external variables x_step, dy and y0).

        :param x:       argument of the step function
        :param module:  module object from which to take the tanh-func (numpy or sympy)
        """

        z1 = tanh_scaling_factor*(x-x_step)  # shift and scale x-axis
        z2 = module.tanh(z1)  # map z1 ∈ (-inf, inf) to interval (-1, 1)
        z3 = (z2 +1 )/2  # map z2 ∈ (-1, 1) to (0, 1)
        z4 = z3*dy + y0  # map z3 ∈ (0, 1) to (y0, y1)

        return z4

    # save docstring and step parameters
    fnc.__doc__ = "approximated step function %f, %f, %f" % (y0, y1, x_step)
    fnc.x_step = x_step
    fnc.y0 = y0
    fnc.y1 = y1
    fnc.tanh_scaling_factor = tanh_scaling_factor

    return fnc



def interval_fnc_factory(x0, x1, y0=0, y1=1, tanh_scaling_factor=1e3):
    """
    Factory to create continuously approximated interval.

    :param x0:       value to trigger y0 -> y1
    :param x1:       value to trigger y1 -> y0
    :param tanh_scaling_factor:
                     scaling of shifted x-value, such that the tanh-func
                     becomes more rectangular for bigger values. Default 1e3

    """
    # tanh maps R to (-1, 1)

    # first map R to (0, 1)
    # then map (0, 1) -> (y0, y1)

    dy = y1-y0

    def fnc(x, x0=x0, x1=x1, module=sp, tanh_scaling_factor=tanh_scaling_factor):

        z1a = tanh_scaling_factor*(x-x0)  # shift and scale x-axis
        z1b = tanh_scaling_factor*(x-x1)  # shift and scale x-axis
        z2a = module.tanh(z1a)  # map z1 ∈ (-inf, inf) to interval (-1, 1)
        z2b = module.tanh(z1b)  # map z1 ∈ (-inf, inf) to interval (-1, 1)
        z3 = (2 + z2a + z2b)/4  # map z2 ∈ (-1, 1) to (0, 1)
        z4 = z3*dy + y0  # map z3 ∈ (0, 1) to (y0, y1)

        return z4

    # save docstring and step parameters
    fnc.__doc__ = f"approximated interval function {x0=}, {x1=}, {y0=}, {y1=}"
    fnc.x0 = x0
    fnc.x1 = x1
    fnc.y0 = y0
    fnc.y1 = y1
    fnc.tanh_scaling_factor = tanh_scaling_factor

    return fnc




interval_01_09 = interval_fnc_factory(0.1, 0.9)
interval_015_05 = interval_fnc_factory(0.15, 0.65)
interval_m015_015 = interval_fnc_factory(-0.15, 0.15, 1, 0)
three_point_element = interval_fnc_factory(-0.1, 0.1, -1, 1)
step_01 = step_factory(0, 1, +0.001, tanh_scaling_factor=1e5)


def relu(x, module=sp):
    return x * step_01(x, module=module)


# TODO: obsolete?
def limit_m1_1(x, module=sp):
    # limit the result between -1 and 1
    cond_low = step_01(x - (-1), module=module)
    cond_high = step_01(x - (1), module=module)

    res = -1*(1-cond_low) + x*cond_low*(1-cond_high) + 1*cond_high

    return res


def limit(x, xmin=-1, xmax=1, module=sp):

    smin = step_01(x - xmin, module=module)
    smax = step_01(xmax - x, module=module)

    return x*smin*smax + xmin*(1-smin) + xmax*(1-smax)


def apx(x, x0, eps=1e-3):
    r"""
    express condition that x \approx x0
    """

    return (sp.Abs(x - x0) < eps)



def create_counter_hyperblock(input_signal, name="countdown", T_storage=1):
    """
    Input must be "on" until the counter has risen > gamma
    """

    # threshold for self sustain
    gamma = .1

    T_fast = .01 * T_storage
    hyperblock = pbs.core.HyperBlock(name=name)
    step_0 = step_factory(0, 1, x_step=0.01)


    _cntr_up, _fb_ed2, = pbs.inputs('_cntr_up, _fb_ed2')


    x1, x2, x3, x4 = pbs.sp.symbols("x1, x2, x3, x4")


    # DT1 Element for input filtering

    z, u = x1, input_signal
    DT1 = pbs.RHSBlock(
        f_expr = 1/(1*T_fast)*(-z + u)*step_0(_cntr_up - gamma),
        h_expr = three_point_element(u-z), # limit
        local_state=(z,),
        insig=u,
    )

    # countdown block (first part), "b1" is a signal name on handwritten notes

    b1 = _fb_ed2
    z, u = x2, sp.Abs(DT1.Y)
    phi = step_0(z-gamma)  # self-sustain
    irCD_1 = pbs.RHSBlock(
        f_expr=(
            # integration speed: value 1 is reached after time T_storage
            1/T_storage*         # slowly loading integrator if ...
                (step_0(1.02-z)                # z < 1
                 *step_0(u + step_0(z-gamma))    # && u==1 || z > .1 (self sustained)
                 *step_0(.1-b1)                # && b==0
            )
            + 1/(T_fast)*(-z)*step_0(b1)  # fast unloading integrator if b1 = 0
        ),

        # !! this is strange

        h_expr=relu(
            (u-z)*(1-phi) + (1-z)*phi # y = relu(u -z) (if z is small) and relu(1-z) (else)
            # u-z
        )*step_0(.02-b1), # but only if b1 == 0
        local_state=(z,),
        insig=u,
    )

    pbs.loop(irCD_1.stateVars[0], _cntr_up)


    # create a rising edge if the countdown is finished
    SUM_LIMIT = pbs.Blockfnc(step_0(step_0(irCD_1.stateVars[0] - 1) + _fb_ed2))


    # hysteresis-based pulse extension (hype)-> b1
    z, u = x3, SUM_LIMIT.Y
    hype = pbs.RHSBlock(
        f_expr=(
            1/(T_fast)*(
                -z                   # PT1 behavior
                + step_0(
                    u + z            # load element if u > 0 or z > 0 (self sustain)
                    - 3*step_0(gamma-irCD_1.stateVars[0]) # overcompensate both inputs on external trigger
                                                        # (countdown state reached 0 again)
                )
            )
        ),
        h_expr=z,
        local_state=(z,),
        insig=u,
    )

    pbs.loop(hype.Y, _fb_ed2)


    # count up
    hyperblock.U = pbs.Blockfnc(irCD_1.stateVars[0]*step_0(.02-b1))


    # sign detection
    z, u = x4, DT1.Y

    hyperblock.SD = pbs.RHSBlock(
        f_expr = 1/T_fast*(-z + u)*step_0(gamma - irCD_1.stateVars[0]),  # store the input when U < gamma
        h_expr = three_point_element(z),
        local_state=(z,),
        insig=u,
    )



    hyperblock.SL = SUM_LIMIT

    # hysteresis-based pulse extension (hype)-> b1
    hyperblock.hype = hype

    # count down
    hyperblock.D = irCD_1
    hyperblock.DT1 = DT1

    # counter active
    hyperblock.A = pbs.Blockfnc(step_0(hyperblock.U.Y))

    # statevars
    hyperblock.sv = [irCD_1.stateVars[0]]

    z = irCD_1.stateVars[0]
    hyperblock.dbg1 = pbs.Blockfnc(1-step_0(z-gamma))

    return hyperblock




def relaxometrie():

    a = 0
    b = 5
    c = 10
    d = 40
    e = 45
    f = 3.35/0.03
    g = 120

    cr1 = 0.6
    cr2 = 0.15

    xr = np.array ([0,5,5.01,5.05,5.06,40,40.01,40.05,40.06,120])
    yr = np.array([0,0,cr1,cr1,0,0,cr2,cr2,0,0])

    if 0:
        plt.plot(xr, yr,'b');

        plt.xlim([0,120])
        plt.grid(True)

        plt.show()

    xr1 = np.linspace(a, b, 50)
    xr2 = np.linspace(b, c, 50)
    xr3 = np.linspace(c, d, 50)
    xr4 = np.linspace(d, e, 50)
    xr5 = np.linspace(e, f, 50)
    xr6 = np.linspace(f, g, 50)

    yr1 = 0*xr1+1
    yr2 = -1+2 / (1 + np.exp(4*xr2-30))
    yr3 = 0.03*xr3-1.3
    #yr4 = -1+0.9 / (1 + 1*np.exp(4*xr4-170))

    tmp = -0.609
    yr4 = yr3[-1] + (1- 1 / (1 + 1*np.exp(4*xr4-170)))*(tmp - yr3[-1])
    yr5 = np.clip(0.03*xr5-2.35 + 1 + tmp, -1, 1)
    yr6 = 0*xr6+1


    style = {"lw": 3, "alpha": 0.5, "ls": "--"}
    plt.plot(xr1, yr1, 'r', **style)
    plt.plot(xr2, yr2, 'r', **style)
    plt.plot(xr3, yr3, 'r', **style)
    plt.plot(xr4, yr4, 'r', **style)
    plt.plot(xr5, yr5, 'r', **style)
    plt.plot(xr6, yr6, 'r', **style)


    if 0:
        plt.axvline(x=5, ymin=0, ymax=1)
        plt.axvline(x=40, ymin=0, ymax=1)

    plt.ylim([-1.1, 1.1])
    plt.xlim([0,120])
    plt.grid(True)



def pain_suppression(dc=None):
    cr1 = 0.3
    cr2 = 0.3

    xs = np.array ([0,5,5.01,5.05,5.06,40,40.01,40.05,40.06,90])
    ys = np.array([0,0,cr1,cr1,0,0,cr2,cr2,0,0])

    if 0:
        # input
        plt.plot(xs, ys,'b');

        plt.xlim([0,90])
        plt.grid(True)

        plt.show()

    aa = 0
    bb = 5
    cc = 10
    dd = 35
    ee = 40
    ff = 45
    gg = 70
    hh = 85
    ii = 90

    xs1 = np.linspace(aa, bb, 50)
    xs2 = np.linspace(bb, cc, 50)
    xs3 = np.linspace(cc, dd, 50)
    xs4 = np.linspace(dd, ee, 50)
    xs5 = np.linspace(ee, ff, 50)
    xs6 = np.linspace(ff, gg, 50)
    xs7 = np.linspace(gg, hh, 50)
    xs8 = np.linspace(hh, ii, 50)

    ys1 = 0*xs1
    ys2 = xs2/10-0.5
    ys3 = 0*xs3+0.5
    ys4 = -xs4/20+2.25
    ys5 = xs5/10-3.75
    ys6 = 0*xs6+0.75
    ys7 = -xs7/20+4.25
    ys8 = 0*xs8

    a = 0
    b = 13.464
    c = 49.368
    d = 90

    xr1 = np.linspace(b, c, 500)
    xr2 = np.linspace(a, b, 50)
    xr3 = np.linspace(c, d, 50)
    yr1 = 0.25*np.sin (0.35*xr1)+0.25
    yr2 = 0*xr2
    yr3 = 0*xr3

    xm2 = np.linspace(b, 37.46, 50)
    ym1 = yr2
    ym2 = 0*xm2
    xm3 = np.linspace(37.46, ee, 50)
    xm4 = np.linspace(ee, 42.06, 50)
    ym3 = (0.25*np.sin (0.35*xm3)+0.25)-(-xm3/20+2.25)
    ym4 = (0.25*np.sin (0.35*xm4)+0.25)-(xm4/10-3.75)
    xm5 = np.linspace(42.06, ii, 50)
    ym5 = 0*xm5

    def plot(style=None):

        if style is None:
            style = {"lw": 3, "alpha": 0.5, "ls": "--"}

        plt.plot(xs1, ys1, 'b', **style)
        plt.plot(xs2, ys2, 'b', **style)
        plt.plot(xs3, ys3, 'b', **style)
        plt.plot(xs4, ys4, 'b', **style)
        plt.plot(xs5, ys5, 'b', **style)
        plt.plot(xs6, ys6, 'b', **style)
        plt.plot(xs7, ys7, 'b', **style)
        plt.plot(xs8, ys8, 'b', **style)

        plt.plot(xr1, yr1, 'r', **style)
        plt.plot(xr2, yr2, 'r', **style)
        plt.plot(xr3, yr3, 'r', **style)


        plt.plot(xr2, ym1, 'g', **style)
        plt.plot(xm2, ym2, 'g', **style)
        plt.plot(xm3, ym3, 'g', **style)
        plt.plot(xm4, ym4, 'g', **style)
        plt.plot(xm5, ym5, 'g', **style)

        # plt.axvline(x=5, ymin=0, ymax=1)
        # plt.axvline(x=40, ymin=0, ymax=1)

        plt.ylim([-0.1, 1.1])
        plt.xlim([0, 90])
        plt.grid(True)

    if dc is not None:
        dc.fetch_locals()


def atropin_dynamics(dc=None):
    """
    Atropin is used to compensate the drop of heart frequency.
    """

    xs = np.array ([0,5,5.01,5.05,5.06,22.5,22.51,22.55,22.6,140])
    ys = np.array([0,0,0.5,0.5,0,0,0,0,0,0])

    xr1 = np.linspace(a, d, 500)
    xr2 = np.linspace(d, e, 50)
    xr3 = np.linspace(e, f, 50)

    yr6 = (40+20 / (1 + np.exp(-4*xr1+25)))
    yr1 = xr3*0+40
    yr2 = -xr2/3+81+2/3

    def plot():

        if 0:

                plt.plot(xs, ys,'b')
                plt.xlim([0,140])
                plt.grid(True)

        plt.plot(xr2, yr2, 'g')
        plt.plot(xr3, yr1, 'g')
        plt.plot(xr1, yr6, 'g')
        plt.axvline(x=5, ymin=0, ymax=1)

        plt.ylim([-0.1, 100])
        plt.xlim([0,140])
        plt.grid(True)

        plt.show()

    if dc is not None:
        dc.fetch_locals()


# this is used to model static dose gain for acrinor
# source: https://stackoverflow.com/a/15196628
def polyfit_with_fixed_points(n, x, y, xf, yf) :
    mat = np.empty((n + 1 + len(xf),) * 2)
    vec = np.empty((n + 1 + len(xf),))
    x_n = x**np.arange(2 * n + 1)[:, None]
    yx_n = np.sum(x_n[:n + 1] * y, axis=1)
    x_n = np.sum(x_n, axis=1)
    idx = np.arange(n + 1) + np.arange(n + 1)[:, None]
    mat[:n + 1, :n + 1] = np.take(x_n, idx)
    xf_n = xf**np.arange(n + 1)[:, None]
    mat[:n + 1, n + 1:] = xf_n / 2
    mat[n + 1:, :n + 1] = xf_n.T
    mat[n + 1:, n + 1:] = 0
    vec[:n + 1] = yx_n
    vec[n + 1:] = yf
    params = np.linalg.solve(mat, vec)
    return params[:n + 1]



def combined_prop_acri_nora_dynamics(dc=None):
    """
    """
    xp = np.array ([1,1.01,20,20.01,40,40.01])
    yp = np.array([0,5,5,8,8,0])

    xs1 = np.array ([3,3.01,3.05,3.06])
    ys1 = np.array([0,2,2,0])
    xs2 = np.array ([10,10.01,10.05,10.1])
    ys2 = np.array([0,2,2,0])
    xs3 = np.array ([25,25.01,43.05,43.06])
    ys3 = np.array([0,7,7,0])

    def plot1():
        plt.figure()
        plt.plot(xp, yp,'b', label = 'Propofol');
        plt.plot(xs1, ys1,'r', label = 'Akrinor');
        plt.plot(xs2, ys2,'r');
        plt.plot(xs3, ys3,'y', label = 'Noradrenalin');

        plt.xlim([0,60])
        plt.grid(True)

        plt.legend()

    # Blutdruckkurve bei isoliertem Propofol-Einfluss

    xr1 = np.linspace(0, 20, 100)
    xr2 = np.linspace(20, 40, 100)
    xr3 = np.linspace(40, 60, 100)
    yr1 = 70+30 / (1 + np.exp(4*xr1-10))
    yr2 = 52+18 / (1 + np.exp(4*xr2-86))
    yr3 = 52+48 / (1 + np.exp(-2*xr3+87 ))


    # Blutdruckkurve bei isoliertem Akrinor-Einfluss

    xq0 = np.linspace(0, 3, 100)
    yq0 = 0*xq0 + 75
    xq1 = np.linspace(3, 10, 100)
    yq1 = 85 - 10*np.exp(-0.5*xq1+1.5)
    xq2 = np.linspace(10, 43, 100)
    yq2 = 95 - 10*np.exp(-0.5*xq2+5.0)
    xq21=np.linspace(43, 50, 100)
    yq21=-xq21+138
    xq22=np.linspace(50, 51.5, 100)
    yq22=-2*xq22+188
    xq3 = np.linspace(51.5, 60, 100)
    yq3 = -xq3+136.5

    # Isolierte Akrinor-Wirkungskurve

    yd0 = yq0 - 75
    yd1 = yq1 - 75
    yd2 = yq2 - 75
    yd21=yq21-75
    yd22=yq22-75
    yd3 = yq3 - 75

    # Isolierte NorA-Wirkungskurve

    xn1 = np.linspace(0, 40, 1000)
    yn1 = 21 / (1 + np.exp(-5*xn1+130))
    xn2 = np.linspace(40, 60, 1000)
    yn2 = 21-21 / (1 + np.exp(-5*xn2+220))

    def plot2():
        plt.figure()
        plt.plot(xr1, yr1, 'b', label = 'Propofol')
        plt.plot(xr2, yr2, 'b')
        plt.plot(xr3, yr3, 'b')

        plt.plot(xq0, yq0, 'r', linestyle = ':')
        plt.plot(xq1, yq1, 'r', linestyle = ':')
        plt.plot(xq2, yq2, 'r', linestyle = ':')
        plt.plot(xq21,yq21,'r', linestyle = ':')
        plt.plot(xq22,yq22,'r', linestyle = ':')
        plt.plot(xq3, yq3, 'r', linestyle = ':')

        plt.plot(xq0, yd0, 'r', label = 'Akrinor')
        plt.plot(xq1, yd1, 'r')
        plt.plot(xq2, yd2, 'r')
        plt.plot(xq21,yd21,'r')
        plt.plot(xq22,yd22,'r')
        plt.plot(xq3, yd3, 'r')

        plt.plot(xn1, yn1, 'y', label = 'Noradrenalin')
        plt.plot(xn2, yn2, 'y')

        plt.ylim([-10, 150])
        plt.xlim([0,60])
        plt.grid(True)

        plt.legend()


    xm1 = np.linspace(0, 3, 100)
    xm2 = np.linspace(3, 10, 100)
    xm2a = np.linspace(10, 20, 100)
    xm3 = np.linspace(20, 40, 100)
    xm4a = np.linspace(40, 43, 100)
    xm4b = np.linspace(43, 50, 100)
    xm4c = np.linspace(50, 51.5, 100)
    xm5 = np.linspace(51.5, 60, 100)

    s1 = 1 #(Noradrenalin)
    s2 = 1 #(Akrinor)

    """
    - Akrinor: 1. Plateauphase endet vor 2.

    """


    ym1 = 70+30 / (1 + np.exp(4*xm1-10))
    ymd1 = ym1 + yd0
    ymd2 = 70+30 / (1 + np.exp(4*xm2-10)) + s2*(10 - 10*np.exp(-0.5*xq1+1.5))
    ymd2a = 70+30 / (1 + np.exp(4*xm2a-10)) + s2*(20 - 10*np.exp(-0.5*xm2a+5.0))
    ym3 = 52+18 / (1 + np.exp(4*xm3-86)) + s2*(20 - 10*np.exp(-0.5*xm3+5.0)) + s1*(21 / (1 + np.exp(-5*xm3+130)))
    ym4a = 52+48 / (1 + np.exp(-2*xm4a+87)) + s2*20 + s1*(21-21 / (1 + np.exp(-5*xm4a+220)))
    ym4b = 52+48 / (1 + np.exp(-2*xm4b+87)) + s2*(-xm4b+138-75) + s1*(21-21 / (1 + np.exp(-5*xm4b+220)))
    ym4c = 52+48 / (1 + np.exp(-2*xm4c+87)) + s2*(-2*xm4c+188-75) + s1*(21-21 / (1 + np.exp(-5*xm4c+220)))
    ym5 = 52+48 / (1 + np.exp(-2*xm5+87)) + s2*(-xm5+136.5-75)


    def plot3():
        plt.figure()

        plt.plot(xm1, ymd1, 'g')
        plt.plot(xm2, ymd2, 'g')
        plt.plot(xm2a, ymd2a, 'g')
        plt.plot(xm3, ym3, 'g')
        plt.plot(xm4a, ym4a, 'g')
        plt.plot(xm4b, ym4b, 'g')
        plt.plot(xm4c, ym4c, 'g')
        plt.plot(xm5, ym5, 'g')

        plt.axvline(x=1, ymin=0, ymax=1)
        plt.axvline(x=20, ymin=0, ymax=1)
        plt.axvline(x=40, ymin=0, ymax=1)
        plt.axvline(x=25, ymin=0, ymax=1)
        plt.axvline(x=43, ymin=0, ymax=1)
        plt.axvline(x=3, ymin=0, ymax=1)
        plt.axvline(x=10, ymin=0, ymax=1)

        plt.ylim([-0.1, 150])
        plt.xlim([0, 60])
        plt.grid(True)

    if dc is not None:
        dc.fetch_locals()


def acrinor_dynamics(dc=None):
    """
    Acrinor is used to compensate the drop of blood pressure.
    """


    a = 0
    b = 5
    c = 7
    d = 15
    e = 22.5
    f = 24.5
    g = 32.5
    h = 45
    i = 50

    cr1 = 2e-2
    cr2 = 2e-2

    t_cr1 = 5
    t_cr2 = 22.5

    xs = np.array ([0,5,5.01,5.05,5.06,22.5,22.51,22.55,22.6,90])
    ys = np.array([0,0,cr1,cr1,0,0,cr2,cr2,0,0])

    xs1 = np.linspace(a, b, 50)
    xs2 = np.linspace(b, c, 50)
    xs3 = np.linspace(c, d, 50)
    xs4 = np.linspace(d, e, 50)
    xs5 = np.linspace(e, f, 50)
    xs6 = np.linspace(f, g, 50)
    xs7 = np.linspace(g, h, 50)
    xs8 = np.linspace(h, i, 50)

    ys1 = 0*xs1+60
    ys2 = 10*xs2+10
    ys3 = 0*xs3+80
    ys4 = -2*xs4+110
    ys5 = 10*xs5-160
    ys6 = 0*xs6+85
    ys7 = -2*xs7+150
    ys8 = 0*xs8+60

    def plot():

        # input
        if 0:
            plt.plot(xs, ys,'b');

            plt.xlim([0,50])
            plt.grid(True)

            plt.show()

        plt.plot(xs1, ys1, 'r')
        plt.plot(xs2, ys2, 'r')
        plt.plot(xs3, ys3, 'r')
        plt.plot(xs4, ys4, 'r')
        plt.plot(xs5, ys5, 'r')
        plt.plot(xs6, ys6, 'r')
        plt.plot(xs7, ys7, 'r')
        plt.plot(xs8, ys8, 'r')

        plt.axvline(x=t_cr1, ymin=0, ymax=1)
        plt.axvline(x=t_cr2, ymin=0, ymax=1)

        plt.ylim([-5, 160])
        plt.xlim([0, 90])
        plt.grid(True)

        plt.show()


    if dc is not None:
        dc.fetch_locals()


def noradrenalin_dynamics(dc=None):
    xn1 = np.linspace(0, 40, 1000)
    yn1 = 21 / (1 + np.exp(-5*xn1+130))
    xn2 = np.linspace(40, 60, 1000)
    yn2 = 21-21 / (1 + np.exp(-5*xn2+220))

    T1 = 25
    T2 = 43

    def plot():

        # input
        xs3 = np.array ([25,25.01,43.05,43.06])
        ys3 = np.array([0,7,7,0])
        plt.plot(xs3, ys3,'y', label = 'Noradrenalin');


        # result

        plt.plot(xn1, yn1, 'y', label = 'Noradrenalin')
        plt.plot(xn2, yn2, 'y')

        plt.ylim([-10, 150])
        plt.xlim([0,60])
        plt.grid(True)



    if dc is not None:
        dc.fetch_locals()



if __name__ == "__main__":
    plt.plot(xp, yp,'b')

    plt.xlim([0,20])
    plt.grid(True)

    plt.figure()

    plt.plot(xr1, yr1, 'r')
    plt.plot(xr2, yr2, 'r')
    plt.plot(xr3, yr3, 'r')
    plt.plot(xr1, yr4, 'g')
    plt.plot(xr2, yr5, 'g')
    plt.plot(xr3, yr6, 'g')
    plt.axvline(x=1, ymin=0, ymax=1)
    plt.axvline(x=5, ymin=0, ymax=1)
    plt.axvline(x=9, ymin=0, ymax=1)

    plt.ylim([-0.1, 150])
    plt.xlim([0,20])
    plt.grid(True)

    plt.figure()


    plt.plot(x1, y1, 'y')
    plt.plot(x2, y2, 'y')
    plt.plot(x3, y3, 'y')
    plt.axvline(x=1, ymin=0, ymax=1)
    plt.axvline(x=5, ymin=0, ymax=1)
    plt.axvline(x=9, ymin=0, ymax=1)

    plt.ylim([-0.1, 1.1])
    plt.xlim([0,20])
    plt.grid(True)

    plt.show()
