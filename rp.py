"""

Computes equilibrium prices and actions using the pricing equation.  The
prices are computed recursively, starting with p(0) = 0, and then working up
to p(1).  Details of the algorithm are given in the paper.

Author: John Stachurski
Last modified: March 2014.

"""


from __init__ import *



class RP(object):
    """
    The base class.   Computes equilibrium prices and actions for the
    functional equation

        p(s) = min_{t <= s} { delta k p(t/k) + c(s - t) }

    In the standard one-dimensional line case, k is set to 1.

    """
    
    def __init__(self, 
            n=1000, 
            delta=1.2, 
            # c=lambda t: np.exp(10*t) - 1,  # TODO: automation level 
            c_list = None,
            breaks = None,
            k=1,
            sbar=1,
            a = 0.0):

        if c_list is None:
            c_list = [lambda s, t, a: np.exp(10*(s-t)) - 1]
            breaks = []

        self._n = n
        self._delta = delta
        # self._c = c
        self._c_list = c_list
        self._breaks = breaks or []
        self._k = k
        self._sbar = sbar
        self._a = a   
        self.update()

    def update(self):
        self.grid = np.linspace(0, self._sbar, num=self._n)
        self.p = np.zeros(self._n)     # Will store vals of p at the grid points
        self.compute_prices()         # Populates p
        self.p_func = lambda x: interp(x, self.grid, self.p)  # Holds the lin interp of p

    def get_a(self): return self._a

    def set_a(self, a): 
        self._a = a
        self.update()
    a = property(get_a, set_a)

    # 兼容包装：既支持 c(x)，也支持 c(s, t, a)
    # def _cost(self, s, t):
    #     t = max(t, 1e-8)  # 避免负数和0
    #     try:
    #         return self._c(s, t, self._a)      # 新式：三参
    #     except TypeError:
    #         return self._c(s - t)              # 旧式：单参

    # ------------- helper: choose segment -----------------
    def _cost(self, s, t):
        """Dispatch to the right cost function according to s."""
        idx = np.searchsorted(self._breaks, s, side='right')
        c   = self._c_list[idx]
        # two possible signatures: c(x)  or  c(s,t,a)
        if c.__code__.co_argcount == 1:          # legacy 1-arg version
            return c(s - t)
        else:                                    # new 3-arg version
            return c(s, t, self._a)

    def get_n(self):
        return self._n

    def get_k(self):
        return self._k

    def get_c(self):
        return self._c_list[0] if self._c_list else None

    def get_delta(self):
        return self._delta

    def get_sbar(self):
        return self._sbar

    def set_n(self, n):
        self._n = n
        self.update()

    def set_k(self, k):
        self._k = k
        self.update()

    def set_c(self, c):
        self._c = c
        self.update()

    def set_delta(self, delta):
        self._delta = delta
        self.update()

    def set_sbar(self, sbar):
        self._sbar = sbar
        self.update()

    n = property(get_n, set_n)
    c = property(get_c, set_c)
    delta = property(get_delta, set_delta)
    sbar = property(get_sbar, set_sbar)
    k = property(get_k, set_k)

    def compute_prices(self, tol=1e-3):
        """
        This is the standard algorithm, involving iteration with T.
        The initial condition is p = c.
        """
        delta, k, n = self._delta, self._k, self._n
        sbar = self._sbar
        # self.p = c(self.grid)  # Initial condition is c(s), as an array
        # self.p = self._c(self.grid) if self._c.__code__.co_argcount == 1 else np.zeros(self._n)
        # initialise: if *all* cost functions are 1-arg, use c(s)
        if all(c.__code__.co_argcount == 1 for c in self._c_list):
            self.p = np.array([self._cost(s, 0) for s in self.grid])
        else:
            self.p.fill(0)

        new_p = np.empty(self.n)
        error = tol + 1
        while error > tol:
            for i, s in enumerate(self.grid):
                p = lambda x: interp(x, self.grid, self.p)
                Tp = lambda t: delta * k * p(t / k) + self._cost(s, t)
                new_p[i] = Tp(fminbound(Tp, 0, s))
            error = np.max(np.abs(self.p - new_p))
            # self.p = new_p
            self.p = new_p.copy()  # 避免引用同一数组

    def compute_prices2(self):
        """
        This is a faster algorithm, although the mathematical justification
        is more involved.
        """
        delta, k, n = self._delta, self._k, self._n
        self.p[0] = 0
        for i in range(1, self.n):
            interp_p = lambda x: interp(x, self.grid[:i], self.p[:i])
            f = lambda t: delta * k * interp_p(t / k) + self._cost(self.grid[i], t)
            self.p[i] = f(fminbound(f, 0, self.grid[i-1]))

    def t_star(self, s):
        """
        Takes p_func as the true function, minimizes on [0,s].  In fact the
        function minimizes on [-1,s] and then takes the max of the minizer and
        zero.  This hack results in better results close to zero.  Thanks to
        Alex Olssen for pointing this out.
        """
        delta, k, n = self._delta, self._k, self._n
        f = lambda t: delta * k * self.p_func(t / k) + self._cost(s, t)
        return max(fminbound(f, -1, s), 0)

    def ell_star(self, s):
        return s - self.t_star(s)





class RPline(RP):
    """
    Subclass for the line model.
    """
    
    def compute_stages(self):
        s = self._sbar
        transaction_stages = [s]
        while s > 0:
            s = self.t_star(s)
            transaction_stages.append(s)
        return transaction_stages

    def plot_prices(self, plottype='-', label=None, plot_stages=False):
        plt.plot(self.grid, self.p,
                plottype, label=label)
        if plot_stages:
            transaction_stages = self.compute_stages()
            for s in transaction_stages:
                plt.axvline(x=s, c="0.5")

    def plot_t_star(self):
        plt.xlim=(0,self.S)
        plt.ylim=(0,self.S)
        plt.plot(self.grid, [self.t_star(s) for s in self.grid], 'b-')

    def plot_ell_star(self):
        plt.xlim=(0,self.S)
        plt.ylim=(0,self.S)
        plt.plot(self.grid, [self.ell_star(s) for s in self.grid], 'b-')



class RPtree(RP):
    """
    Subclass for the tree model (k > 1).
    """
    def compute_stages(self):
        """
        This function creates a list called levels, where levels[n] is a tuple
        (t_n, l_n, v_n).  The value n is the level in the tree, with the first firm
        (i.e., most downstream) at level 0.  An n-th level firm receives a contract
        of size t_n (amount of tasks to be completed) and chooses an amount l_n
        to complete in-house.  The quantity v_n is the value added of a level n
        firm.
        """
        levels = []
        t = 1
        while t > 0:
            l = self.ell_star(t)
            t_next = (t - l) / self.k
            v = self.p_func(t)  - self.k * self.p_func(t_next)
            levels.append((t, l, v))
            t = t_next
        return levels

    def plot_prices(self, plottype='-', label=None):
        plt.plot(self.grid, self.p, plottype, label=label)


##
#############   An example of usage   ####################
##


if __name__ == '__main__':

    if 1:
        ps = RPline()
        ts = ps.compute_stages()
        #print "Number of firms:", len(ts)
        ps.plot_prices(plot_stages=True)
        plt.show()

    if 0:
        ps = RPtree(k=2)
        levels = ps.compute_stages()
        ts, ls, vs = [], [], []
        for level in levels:
            t, l, v = level
            ts.append(t)
            ls.append(l)
            vs.append(v)
        # Count the number of firms
        num_firms = 0
        for n in range(len(levels)):
            num_firms += ps.k**n
        #print "Number of firms:", num_firms
        ps.plot_prices()
        plt.show()

