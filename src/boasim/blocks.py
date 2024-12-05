from typing import List
from string import Template

import sympy as sp
import numpy as np
from sympy.utilities.lambdify import implemented_function

import symbtools as st
from pyblocksim.td import TDBlock, new_TDBlock, T, t, k, limit, sign, dtDelay1, eq

class dtDirectionSensitiveSigmoid(new_TDBlock(5)):
    """
    This Sigmoid behaves differently for positive and negative input steps
    """

    def rhs(self, k: int, state: List) -> List:

        assert "K" in self.params
        assert "T_trans_pos" in self.params # overall counter time
        assert "T_trans_neg" in self.params
        assert "sens" in self.params

        # fraction of overall counter time that is used for waiting
        f_wait_pos = getattr(self, "f_wait_pos", 0)
        f_wait_neg = getattr(self, "f_wait_neg", 0)

        assert 0 <= f_wait_pos <= 1
        assert 0 <= f_wait_neg <= 1

        x1, x2, x_cntr, x_u_storage, x_debug  = self.state_vars
        x_u_storage_new = self.u1

        # determine a change of the input
        input_change = sp.Piecewise((1, sp.Abs(self.u1 - x_u_storage) > self.sens), (0, True))

        # counter
        pos_delta_cntr = T/self.T_trans_pos
        neg_delta_cntr = -T/self.T_trans_neg
        x_cntr_new =  sp.Piecewise(
            (pos_delta_cntr, self.u1 - x_u_storage > self.sens),
            (neg_delta_cntr, self.u1 - x_u_storage < - self.sens),
            # note that expressions like 0 < x < 1 are not possible for sympy symbols
            (x_cntr + pos_delta_cntr, (0 < x_cntr) & (x_cntr<= 1)),
            (x_cntr + neg_delta_cntr, (0 < -x_cntr) & (-x_cntr<= 1)),
            (0, True),
        )

        # implement the waiting
        # effective waiting fraction
        f_wait = sp.Piecewise((f_wait_neg, x_cntr < 0), (f_wait_pos, x_cntr > 0), (0, True))

        # effective counter (reaching the goal early by intention)
        x_cntr_eff = limit(sp.Abs(x_cntr), xmin=f_wait, xmax=.95, ymin=0, ymax=1)*sign(x_cntr)

        T_fast = 2*T

        # this will reach 0 before |x_cntr_eff| will reach 1
        # count_down = limit(1-1.2*sp.Abs(x_cntr_eff), xmin=0, xmax=1, ymin=0, ymax=1)
        count_down = 1 - sp.Abs(x_cntr_eff)


        T_trans = sp.Piecewise((self.T_trans_neg, x_cntr < 0), (self.T_trans_pos, x_cntr > 0), (0, True) )

        # !! self muss raus
        T1 = T_fast + .6*self.T_trans_pos*(1+40*count_down**10)/12

        p12 = 0.6

        phase0 = sp.Piecewise((1, sp.Abs(x_cntr_eff) <= 1e-4), (0, True))
        phase2 = limit(sp.Abs(x_cntr_eff), xmin=p12, xmax=1, ymin=0, ymax=1)*(1-phase0)
        phase1 = (1 - phase2)*(1-phase0)


        x_debug_new = x_cntr_eff#  phase0*10 + phase1
        # x_debug_new = T1
        # x_debug_new = self.u1 - x_u_storage

        # PT2 Element based on Euler forward approximation
        x1_new = sum((
            x1,
            (T*x2)*phase1,    # ordinary PT2 part
            (T/T_fast*(self.K*self.u1 - x1))*phase2,  # fast PT1-convergence towards the stationary value
        ))

        # at the very end we want x1 == K*u1 (exactly)
        x1_new = sp.Piecewise((x1_new, sp.Abs(x_cntr)<= 1), (self.K*self.u1, True))


        # x2 should go to zero at the end of transition
        x2_new = sum((
            x2*phase1,
            T*(1/(T1*T1)*(-(T1 + T1)*x2 - x1 + self.K*self.u1))*phase1,

        ))

        return [x1_new, x2_new, x_cntr_new, x_u_storage_new, x_debug_new]


class dtRelaxoBlock(new_TDBlock(5)):
    """
    This block models relaxometrics
    """

    def rhs(self, k: int, state: List) -> List:

        assert "sens" in self.params
        assert "K" in self.params
        assert "slope" in self.params
        # assumption: next nonzero input not in sigmoid phase

        x1, x2, x_cntr, x4_buffer, x_debug  = self.state_vars

        # time for sigmoid phase
        self.T_phase1 = 5


        # counter
        pos_delta_cntr = T/self.T_phase1
        x_cntr_new =  sp.Piecewise(
            (pos_delta_cntr, self.u1 > 0),
            (x_cntr + pos_delta_cntr, (0 < x_cntr) & (x_cntr<= 1)),
            (0, True),
        )

        sigmoid_phase_cond = (self.u1 > 0) | ((0 < x_cntr) & (x_cntr<= 1))

        # if sigmoid phase is over (counter reached 1) x4 := x1
        x4_buffer_new = sp.Piecewise((x4_buffer + self.K*self.u1, sigmoid_phase_cond), (x1, True))

        # effective counter (here: same as normal)
        x_cntr_eff = x_cntr

        T_fast = 2*T
        count_down = 1 - sp.Abs(x_cntr_eff)


        T_trans = sp.Piecewise((self.T_phase1, x_cntr > 0), (0, True) )

        T1 = T_fast + .6*T_trans*(1+40*count_down**10)/12

        p12 = 0.6


        counter_active = sp.Piecewise((1, x_cntr > 0), (0, True))

        phase0 = 0# sp.Piecewise((1, sp.Abs(x_cntr_eff) <= 1e-4), (0, True))
        phase2 = 0# limit(sp.Abs(x_cntr_eff), xmin=p12, xmax=1, ymin=0, ymax=1)*(1-phase0)
        phase1 = counter_active


        x_debug_new = x_cntr_eff#  phase0*10 + phase1
        # x_debug_new = T1
        # x_debug_new = self.u1 - x_u_storage

        # PT2 Element based on Euler forward approximation (but here x4 is the "input")

        v = x4_buffer

        x1_new = sum((
            x1,
            (T*x2)*counter_active,   # ordinary PT2 part

            # decay with constant rate
            (self.slope)*sp.Piecewise((1, x1 > 0), (0, True))*(1-counter_active)  # assumption: slope < 0
        ))

        # at the very end we want x1 == K*u1 (exactly)
        x1_new = sp.Piecewise((x1_new, sp.Abs(x_cntr)<= 1), (v, True))

        # x2 should go to zero at the end of transition
        x2_new = sum((
            x2*phase1,
            T*(1/(T1*T1)*(-(T1 + T1)*x2 - x1 + v))*phase1,
        ))

        return [x1_new, x2_new, x_cntr_new, x4_buffer_new, x_debug_new]

    def output(self):
        return 1 - self.x1


class dtSufenta(new_TDBlock(5)):
    """
    This block models pain suppression with Sufentanil
    """

    def rhs(self, k: int, state: List) -> List:

        assert "rise_time" in self.params
        assert "down_slope" in self.params
        assert self.down_slope < 0

        # how long effect stays constant (dependent on input)
        assert "active_time_coeff" in self.params
        assert "dose_gain" in self.params

        # assumption: next nonzero input not in rising or const phase

        x1, x2_target_effect, x3_cntr, x4_slope, x_debug  = self.state_vars

        # value by which the counter is increased in every step
        # after N = rise_time/T steps the counter reaches 1
        delta_cntr1 = T/self.rise_time

        eps = 1e-8 # prevent ZeroDivisionError when calculating unused intermediate result
        delta_cntr2 = sp.Piecewise(
            (T/(self.active_time_coeff*(x2_target_effect/self.dose_gain + eps)), x2_target_effect > 0),
            (0, True)
        )
        x3_cntr_new =  sp.Piecewise(
            (delta_cntr1, self.u1 > 0),
            (x3_cntr + delta_cntr1, (0 < x3_cntr) & (x3_cntr<= 1)),
            # after counter reached 1 -> count from 1 to 2
            (x3_cntr + delta_cntr2, (1 < x3_cntr) & (x3_cntr<= 2)),
            (x3_cntr, (2 < x3_cntr) & (x3_cntr<= 3) & (x1 > 0)),
            (0, True),
        )


        # save uninfluenced target effect for input dose if it is unequal zero
        # "uninfluenced" means as if it was starting from zero
        # -> this yields the correct slope and active time
        # however the real value of x1 might be higher, if the 2nd input dose comes for x1 > 0
        # then x1 is simply increased by the slope
        x2_target_effect_new = sp.Piecewise(
            (x1*0+ self.u1*self.dose_gain, self.u1 > 0),
            (x2_target_effect, True),
        )

        x4_slope_new = sp.Piecewise(
            (x2_target_effect_new/self.rise_time*T, self.u1 > 0),
            (x4_slope, True),
        )

        x1_new = sp.Piecewise(
            (x1 + x4_slope_new, (0 < x3_cntr) & (x3_cntr<= 1)),
            (x1, (1 < x3_cntr) & (x3_cntr<= 2)),
            (x1 + T*self.down_slope, (2 < x3_cntr) & (x3_cntr<= 3) & (x1 >  T*self.down_slope)),
            (0, True),
        )

        x_debug_new = 0

        res = [x1_new, x2_target_effect_new, x3_cntr_new, x4_slope_new, x_debug_new]
        return res


    def output(self):
        return self.x1


class MaxBlockMixin:
    """
    Allows to calculate the maximum of a length3-sequence of expressions
    """
    def _define_max3_func(self):
        cached_func = self._implemented_functions.get("max3_func")
        if cached_func is not None:
            return cached_func

        def max3_func_imp(a, b, c):
            return max(a, b, c)

        max3_func = implemented_function("max3_func", max3_func_imp)

        # the following is necessary for fast implementation
        max3_func.c_implementation = Template("""
            double max3_func(double a, double b, double c) {
                double v1, v2;
                v1 = fmax(a, b);
                v2 = fmax(b, c);
                return fmax(v1, v2);
            }
        """).substitute()

        self._implemented_functions["max3_func"] = max3_func
        return max3_func


class CounterBlockMixin:
    def _define_counter_func_1state(self):
        """
        This is for a 1-state counter (it is started in the rhs function) and just counts down
        """

        cached_func = self._implemented_functions.get("counter_func_2state")
        if cached_func is not None:
            return cached_func

        def counter_func_1state_imp(counter_state, counter_index_state, i, initial_value):
            """
            :param counter_state:   float; current value of the counter
            :param counter_index_state:
                                    int: index which counter should be activated next (was not active)
                                    (allows to cycle through the counters)
            :param i:               int; index which counter is currently considered in the counter-loop
            :param initial_value:   float; value with which the counter is initialized
            """
            # check if the counter-loop (i) is considering the counter which should be activated next
            if counter_index_state == i and initial_value > 0:
                # activate this counter

                assert counter_state == 0
                return initial_value
            if counter_state > 0:
                return counter_state - 1

            return 0
        # convert the python-function into a applicable sympy function
        counter_func_1state = implemented_function("counter_func_1state", counter_func_1state_imp)

        # the following is necessary for fast implementation
        counter_func_1state.c_implementation = Template("""
            double counter_func_1state(double counter_state, double counter_index_state, double i, double initial_value) {
                double result;
                double down_slope = $down_slope;
                if ((counter_index_state == i) && (initial_value > 0)) {
                    // assign the initial value to the counter_state
                    return initial_value;
                }

                // check if the counter (i) is currently running
                if (counter_state > 0) {

                    // counter is assumed to be counting down, thus down_slope is < 0
                    result = counter_state + down_slope;
                    if (result < 0) {
                        result = 0;
                    }
                    return result;
                }

                // if the counter is not running, return 0
                return 0;
            }
        """).substitute(down_slope=-1)

        self._implemented_functions["counter_func_1state"] = counter_func_1state
        return counter_func_1state

    def _define_counter_func_2state(self):
        """
        This is for a 2-state counter (which starts not now but in somewhere in the future)
        """

        cached_func = self._implemented_functions.get("counter_func_2state")
        if cached_func is not None:
            return cached_func

        def counter_func_2state_imp(counter_state, counter_k_start, k, counter_index_state, i, initial_value):
            """
            :param counter_state:   float; current value of the counter
            :param counter_k_start: int; time step index when this counter started
            :param k:               int; current time step index
            :param counter_index_state:
                                    int: index which counter should be activated next (was not yet active)
                                    (allows to cycle through the counters)
            :param i:               int; index which counter is currently considered in the counter-loop
            :param initial_value:   float; value with which the counter is initialized
            """
            # check if the counter-loop (i) is considering the counter which should be activated next
            if counter_index_state == i and initial_value > 0:

                # if counter state is newly loaded it should be zero before
                # print(f"{k=}, new iv: {initial_value}")
                assert counter_state == 0

                # assign the initial value to the counter_state
                return initial_value

            # check if the counter (i) is currently running
            if (counter_state > 0) and (k >= counter_k_start):

                # counter is assumed to be counting down, thus down_slope is < 0
                res = counter_state + self.down_slope
                if res < 0:
                    res = 0
                return res

            # if the counter is not running, do not change the state
            return counter_state

        # convert the python-function into a applicable sympy function
        counter_func_2state = implemented_function("counter_func_2state", counter_func_2state_imp)

        # the following is necessary for fast implementation
        counter_func_2state.c_implementation = Template("""
            double counter_func_2state(double counter_state, double counter_k_start, double k, double counter_index_state, double i, double initial_value) {
                double result;
                double down_slope = $down_slope;
                if ((counter_index_state == i) && (initial_value > 0)) {
                    // assign the initial value to the counter_state
                    return initial_value;
                }

                // check if the counter (i) is currently running
                if ((counter_state > 0) && (k >= counter_k_start)) {

                    // counter is assumed to be counting down, thus down_slope is < 0
                    result = counter_state + down_slope;
                    if (result < 0) {
                        result = 0;
                    }
                    return result;
                }

                // if the counter is not running, do not change the state
                return counter_state;
            }
        """).substitute(down_slope=self.down_slope)

        self._implemented_functions["counter_func_2state"] = counter_func_2state
        return counter_func_2state

    def _define_counter_start_func_2state(self, delta_k):
        ":param delta_k: the amount of time_steps in the future when the counter will start"

        cached_func = self._implemented_functions.get("counter_start_func_2state")
        if cached_func is not None:
            return cached_func


        def counter_start_func_2state_imp(counter_k_start, k, counter_index_state, i, initial_value):
            """
            :param counter_k_start: int; time step index when this counter started
            :param k:               int; current time step index
            :param counter_index_state:
                                    int: index which counter should be activated next
                                    (allows to cycle through the counters)
            :param i:               int; index which counter is currently considered in the counter-loop
            :param initial_value:   float; value with which the counter is initialized
            """

            if counter_index_state == i and initial_value > 0:
                # the counter k_start should be set
                return k + delta_k

            # change nothing
            return counter_k_start

        counter_start_func_2state = implemented_function(f"counter_start_func_2state", counter_start_func_2state_imp)
        counter_start_func_2state.c_implementation = Template("""

            double counter_start_func_2state(double counter_k_start, double k, double counter_index_state, double i, double initial_value) {
                double delta_k = $delta_k;
                if ((counter_index_state == i) && (initial_value > 0)) {
                    // the counter k_start should be set
                    return k + delta_k;
                }

                // change nothing
                return counter_k_start;
            }


        """).substitute(delta_k=delta_k)

        self._implemented_functions["counter_start_func_2state"] = counter_start_func_2state

        return counter_start_func_2state


# This determines how many overlapping Akrinor bolus doses can be modelled
# Should be increased to 10
N_akrinor_counters = 3
class dtAkrinor(new_TDBlock(5 + N_akrinor_counters*2), CounterBlockMixin):
    """
    This block models blood pressure increase due to Akrinor
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.non_counter_states = self.state_vars[:len(self.state_vars)-2*N_akrinor_counters]
        self.counter_state_vars = self.state_vars[len(self.state_vars)-2*N_akrinor_counters:]

        self.counter_states = self.counter_state_vars
        self.n_counters = N_akrinor_counters

    def rhs(self, k: int, state: List) -> List:

        # time to exponentially rise 75%
        assert "T_75" in self.params
        assert "T_plateau" in self.params
        assert "body_mass" in self.params
        assert "down_slope" in self.params
        assert self.down_slope < 0

        # gain in mmHg/(ml/kgG)
        # TODO: this has to be calculated as percentage (see red Text in pptx)
        assert "dose_gain" in self.params

        # assume self.u2 is the current system MAP
        assert isinstance(self.u2, sp.Expr)

        # assumptions:
        # next nonzero input not in rising phase
        # if next nonzero input in plateau or down_slope phase this has to be handled differently (second element)

        x1, x2_integrator, x3_PT1, x4_counter_idx, x5_debug  = self.non_counter_states

        # conventional time constant for exponential rising
        T1 = self.T_75/np.log(4)

        # absolute_map_increase will be the plateau value of the curve
        # absolute_map_increase must be calculated according characteristic curve and current MAP
        # u1: dose of current bolus, u2: current MAP
        absolute_map_increase = self.u1*self.dose_gain/self.body_mass*self.u2

        """
        The counter mechanism works like this:

        - Every counter is associated with two scalar states
        - c0 is associated with self.counter_states[0] and self.counter_states[1]
        - self.counter_states[0]
        - self.counter_states[1]
        - all counters start inactive; if c0 gets activated self.counter_states[0] gets a nonzero value
        - in every time step: all counter_states have to be updated, because of the paradigm:
            new_total_state := state_func(current_total_state)
        - if in time step k the input (`initial_value`) is non-zero the counter which is associated with
            counter_index_state gets prepared. More precisely two things happen (assuming c0 is the one):
            - `counter_func_2state` -> load initial value into self.counter_states[0]
            - `counter_start_func_2state` -> load the index into self.counter_states[1] at which the counter
                actually will start to count down (after T_plateau is over)
        """
        # create/restore functions for handling the counters
        counter_func_2state = self._define_counter_func_2state()

        delta_k = int(self.T_plateau/T)
        counter_start_func_2state = self._define_counter_start_func_2state(delta_k=delta_k)

        # this acts as the integrator
        counter_sum = 0

        new_counter_states = [None]*len(self.counter_states)
        # the counter-loop
        for i in range(self.n_counters):

            # counter_value for index i
            new_counter_states[2*i] = counter_func_2state(
                self.counter_states[2*i], self.counter_states[2*i + 1], k, x4_counter_idx, i, absolute_map_increase
            )

            counter_sum += new_counter_states[2*i]

            # k_start value for index i
            new_counter_states[2*i + 1] = counter_start_func_2state(
                self.counter_states[2*i + 1], k, x4_counter_idx, i, absolute_map_increase
            )

        # T1: time constant of PT1 element, e1: factor for time discrete PT1 element
        e1 = np.exp(-T/T1)

        # old: PT1-like decreasing
        # x1_new = x3_PT1

        # new: PT1-like increasing, linear decreasing
        # (this requires the monkey-patch for is_constant (see above))
        x1_new = sp.Piecewise((x3_PT1, x3_PT1 < counter_sum), (counter_sum, True))

        # currently not used
        x2_new = 0 # x2_integrator + absolute_map_increase

        # counter_sum serves as input signal with stepwise increase and linear decrease
        x3_new = e1*x3_PT1 + 1*(1-e1)*counter_sum

        # increase the counter index for every nonzero input, but start at 0 again
        # if all counters have been used
        x4_new = (x4_counter_idx + sp.Piecewise((1, absolute_map_increase >0), (0, True))) % self.n_counters
        x5_debug_new = 0 # debug_func(self.u1 > 0, k, self.u2, "k,u2")

        res = [x1_new, x2_new, x3_new, x4_new, x5_debug_new] + new_counter_states
        return res

    def output(self):
        return self.x1

N_propofol_counters = 3
class dtPropofolBolus(new_TDBlock(7 + 3*N_propofol_counters), CounterBlockMixin, MaxBlockMixin):
    """
    This block models blood pressure increase due to Propofol bolus doses.
    It uses 1state counters. Each counter, is followed by associated auxiliary values:
    i + 0: counter
    i + 1: effect amplitude considering sensitivity (used for BIS)
    i + 2: effect amplitude without considering sensitivity (used for HR/BP)
    (also considered part of the counter state components ("counter states"))

    """

    def __init__(self: TDBlock, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.n_inputs == 2

        self.non_counter_states = self.state_vars[:len(self.state_vars)-3*N_propofol_counters]
        self.counter_states = self.state_vars[len(self.state_vars)-3*N_propofol_counters:]

        self.n_counters = N_propofol_counters

        # this is set via the params-mechanism of TDBlock
        assert isinstance(self.reference_bp, (int, float))
        assert isinstance(self.reference_bis, (int, float))

        # Note: counter is used both for effect and sensitivity
        self.T_counter = 6

        self.propofol_bolus_sensitivity_dynamics_np = st.expr_to_func(
            t, self.propofol_bolus_sensitivity_dynamics(t), modules="numpy"
        )

        dose = sp.Symbol("dose")
        self.propofol_bolus_static_values_np = st.expr_to_func(
            dose, self.propofol_bolus_static_values(dose), modules="numpy"
        )

        self.effect_dynamics_expr = self._generate_effect_dynamics_expr()

    def _generate_effect_dynamics_expr(self):
        r"""
        Generate a piecewise defined polynomial like: _/‾‾\_ (amplitude 1) ("hat function").
        Used for MAP and BIS.
        """
        Ta = 2
        Tb = 4
        Tc = 6

        static_value = 1

        # rising part
        poly1 = st.condition_poly(t, (0, 0, 0, 0), (Ta, static_value, 0, 0)) ##:

        # falling part
        poly2 = st.condition_poly(t, (Tb, static_value, 0, 0), (Tc, 0, 0, 0)) ##:

        effect_dynamics_expr = sp.Piecewise(
            (0, t < 0), (poly1, t <= Ta), (static_value, t <= Tb), (poly2, t <= Tc), (0, True)
        )
        return effect_dynamics_expr

    def rhs(self, k: int, state: List) -> List:

        (
            x1_bp_effect,
            x2_sensitivity,
            x3_c_ppf_ib_bp,
            x4_counter_idx,
            x5_c_ppf_ib_bis,
            x6_bis_effect,
            x7_debug,
        ) = self.non_counter_states

        # meanings:

        # x1_bp_effect      total blood pressure effect
        # x2_sensitivity    sensitivity \in [1, 1.3], needed for bis
        # x3_c_ppf_ib_bp    current propofol in blood (blood pressure related, without sens)
        # x5_c_ppf_ib_bis   current propofol in blood (BIS related, without sens)

        # the following state components are not really used (but help development)
        # x1_bp_effect
        # x3_c_ppf_ib_bp
        # x5_c_ppf_ib_bis

        # the counter should start immediately -> 1 state version
        # create/restore functions for handling the counters
        counter_func_1state = self._define_counter_func_1state()

        # TODO: make this better parameterizable
        counter_max_val = self.T_counter/T
        counter_target_val = sp.Piecewise((counter_max_val, self.u1 > 0), (0, True))

        new_counter_states = [None]*len(self.counter_states)
        partial_sensitivities = [None]*self.n_counters
        partial_dynamic_dose__bis = [None]*self.n_counters
        partial_dynamic_dose__bp = [None]*self.n_counters

        # amplitude_func = self._define_amplitude_func() # (self.u1, *self.counter_states)

        # the counter-loop
        # Explanation: For counter index i `self.counter_states[3*i]` is the counter value
        # and `self.counter_states[3*i + j]` are the current partial doses (for map (=bp) and bis)

        # new partial doses are mostly 0, except when there is a nonzero input
        # The BIS is affected by the sensitivity effect (`x2`)
        new_partial_dose__bis = self.u1*x2_sensitivity

        # The MAP (bp) is not affected by the sensitivity effect
        new_partial_dose__bp = self.u1

        for i in range(self.n_counters):

            # counter_value for index i
            counter_i = new_counter_states[3*i] = counter_func_1state(
                self.counter_states[3*i], x4_counter_idx, i, counter_target_val
            )

            counter_time = sp.Piecewise(((counter_max_val - counter_i)*T, counter_i > 0), (0, True))
            partial_sensitivities[i] = self.propofol_bolus_sensitivity_dynamics(counter_time)

            current_counter = self.counter_states[3*i]

            # calculate the BIS amplitude (`new_counter_states[3*i + 1]`)
            current_partial_dose__bis = self.counter_states[3*i + 1]
            new_counter_states[3*i + 1] = sp.Piecewise(
                (new_partial_dose__bis, eq(i, x4_counter_idx)), (0,  eq(current_counter, 0)), (current_partial_dose__bis, True)
            )
            partial_dynamic_dose__bis[i] = self._single_dose_effect_dynamics(counter_time, current_partial_dose__bis)
            # BIS is done

            # calculate the BP amplitude (`new_counter_states[3*i + 2]`)
            # Update: this is now only the bp-relevant propofol dose in blood
            current_partial_dose__bp = self.counter_states[3*i + 2]

            # save the amplitude for the next step
            new_counter_states[3*i + 2] = sp.Piecewise(
                (new_partial_dose__bp, eq(i, x4_counter_idx)), (0,  eq(current_counter, 0)), (current_partial_dose__bp, True)
            )
            partial_dynamic_dose__bp[i] = self._single_dose_effect_dynamics(counter_time, current_partial_dose__bp)
            # MAP is done

        # increase the counter index for every nonzero input, but start at 0 again
        # if all counters have been used (achieved by modulo (%))
        x4_counter_idx_new = (x4_counter_idx + sp.Piecewise((1, self.u1 > 0), (0, True))) % self.n_counters

        max3_func = self._define_max3_func()
        assert self.n_counters == 3
        x2_bis_sensitivity_new = max3_func(*partial_sensitivities)

        # IPS()
        cumulated_dynamic_dose__bp = sum(partial_dynamic_dose__bp)
        cumulated_dynamic_dose__bis = sum(partial_dynamic_dose__bis)
        x1_bp_effect_new = (
            -1 * self.propofol_bolus_static_values(cumulated_dynamic_dose__bp) * self.reference_bp * 0.5
        )

        x3_c_ppf_ib_bp_new = cumulated_dynamic_dose__bp

        x5_c_ppf_ib_bis_new = cumulated_dynamic_dose__bis
        x6_bis_effect_new = (
            -1 * self.propofol_bolus_static_values(cumulated_dynamic_dose__bis) * self.reference_bis
        )

        # for debugging
        x7_debug_new = sum(partial_dynamic_dose__bp)

        new_state = [
            x1_bp_effect_new,
            x2_bis_sensitivity_new,
            x3_c_ppf_ib_bp_new,
            x4_counter_idx_new,
            x5_c_ppf_ib_bis_new,
            x6_bis_effect_new,
            x7_debug_new,
        ] + new_counter_states

        return new_state

    def _single_dose_effect_dynamics(self, counter_time, amplitude):

        res = self.effect_dynamics_expr.subs(t, counter_time)*amplitude
        return res

    def output(self):
        res = sp.Matrix([self.x1, self.x6])

        return res

    def propofol_bolus_sensitivity_dynamics(self, t):
        """
        :param t:    time since last bolus (sympy expression)
        """

        k = 0.52175
        maxval = 1.26

        t_peak = 1.5

        f1 = - k / (2 + sp.exp(10*t - 5)) + maxval
        f2 = k / (2 + sp.exp(2.5*t - 8)) + 1

        res = sp.Piecewise((f1, t < t_peak), (f2, True))

        return res

    def propofol_bolus_static_values(self, dose: float):
        """
        :param dose:    specific dose in mg/kgBW
        :returns:       effect_of_medication (between 0 and 1)

        """
        k = -0.34655
        effect_of_medication = 1 - sp.exp(k*dose)
        return effect_of_medication


class PropofolCont(dtDirectionSensitiveSigmoid):
    def _class_specific_params(self):
        return {
            "T_trans_pos": 3,
            "T_trans_neg": 9,
            "K": 1,
            "sens": 0.1,  # sensitivity to detect input change
        }

    def output(self):

        # rr0 = 100 # Ausgangsblutdruck zum Zeitpunkt 0
        # fp = 6 # Volumenstatus und kardiale Kompensationsfähigkeit: 0 (gut) bis 10 (schlecht)

        res = sp.Matrix([
            self.rr0 - self.x1*self.fp,  # MAP
            self.hf0 - self.x1*self.fp   # HR
        ])
        return res

"""

u_amplitude = 10
u_step_time = 1
T_trans_pos = 3
T_trans_neg = 9
T = pbs.td.T



# bp_static  = pbs.Blockfnc(oc.rr0 - DAPT2.Y*oc.fp)
# hf_static  = pbs.Blockfnc(oc.hf0 - DAPT2.Y*oc.fp)





class HF(pbs.td.dtDirectionSensitiveSigmoid):
    def output(self):
        return oc.hf0 - self.x1*oc.fp

params = dict(K=1, T_trans_pos=T_trans_pos, T_trans_neg=T_trans_neg, sens=.1, f_wait_neg=3/9)

u1_expr = sp.Piecewise((0, t < 1), (5, t < T_trans_pos + 2), (8, t < 2*T_trans_pos + 3), (0, True))
ufunc = st.expr_to_func(t, u1_expr)

# dtsigm_1 = pbs.td.dtSigmoid(input1=u1_expr, params=dict(K=1, T_trans=T_trans, sens=.1))

bp = BP(input1=u1_expr, params=params)
hf = HF(input1=u1_expr, params=params)

kk, xx, bo = pbs.td.blocksimulation(250)


"""
