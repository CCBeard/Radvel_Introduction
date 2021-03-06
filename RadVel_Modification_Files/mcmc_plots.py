import numpy as np
import corner
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as pl
from matplotlib import rcParams

from radvel import plot

"""
Module for plotting results of MCMC analysis, including:
    - trend plot
    - autocorrelation plot
    - corner plot of fitted parameters
    - corner plot of derived parameters
"""


class TrendPlot(object):
    """
    Class to handle the creation of a trend plot to show
    the evolution of the MCMC as a function of step number.

    Args:
        post (radvel.Posterior): Radvel Posterior object
        chains (DataFrame): MCMC chains output by radvel.mcmc
        nwalkers (int): number of walkers used in this particular MCMC run
        outfile (string [optional]): name of output multi-page PDF file

    """

    def __init__(self, post, chains, nwalkers, nensembles, outfile=None):

        self.chains = chains
        self.outfile = outfile
        self.nwalkers = nwalkers
        self.nensembles = nensembles

        self.labels = sorted([k for k in post.params.keys() if post.params[k].vary])
        self.texlabels = [post.params.tex_labels().get(l, l) for l in self.labels]
        #Added by CB 04/18/2022
        if 'gp_amp0' in post.params.keys():
            for i in range(len(self.texlabels)):
                if self.texlabels[i] == '$\\eta_{1}_{a}$':
                    self.texlabels[i] = '$\\eta_{0}$'
                else:
                    continue
        self.colors = [plot.cmap(x) for x in np.linspace(0.05, 0.95, nwalkers)]

    def plot(self):
        """
        Make and save the trend plot as PDF
        """
        with PdfPages(self.outfile) as pdf:
            for param, tex in zip(self.labels, self.texlabels):
                flatchain = self.chains[param].values
                wchain = flatchain.reshape((self.nwalkers, self.nensembles, -1))

                _ = pl.figure(figsize=(18, 10))
                for w in range(self.nwalkers):
                    for e in range(self.nensembles):
                        pl.plot(
                            wchain[w][e], '.', rasterized=True, color=self.colors[w],
                            markersize=4
                        )

                pl.xlim(0, wchain.shape[2])

                pl.xlabel('Step Number')
                try:
                    pl.ylabel(tex)
                except ValueError:
                    pl.ylabel(param)

                ax = pl.gca()
                ax.set_rasterized(True)

                pdf.savefig()
                pl.close()

        print("Trend plot saved to %s" % self.outfile)


class AutoPlot(object):
    """
    Class to handle the creation of an autocorrelation time plot
    from output autocorrelation times.

    Args:
        auto (DataFrame): Autocorrelation times output by radvel.mcmc
        saveplot (str, optional): Name of output file, will show as
            interactive matplotlib window if not defined.

    """
    def __init__(self, auto, saveplot=None):

        self.auto = auto
        self.saveplot = saveplot

    def plot(self):
        """
        Make and either save or display the autocorrelation plot
        """

        fig = pl.figure(figsize=(6, 4))
        pl.scatter(self.auto['autosamples'], self.auto['automin'], color = 'blue', label='Minimum Autocorrelation Time')
        pl.scatter(self.auto['autosamples'], self.auto['automean'], color = 'black', label='Mean Autocorrelation Time')
        pl.scatter(self.auto['autosamples'], self.auto['automax'], color = 'red', label='Maximum Autocorrelation Time')
        pl.plot(self.auto['autosamples'], self.auto['autosamples']/self.auto['factor'][0], linestyle=':', color='gray',
                label='Autocorrelation Factor Criterion (N/{})'.format(self.auto['factor'][0]))
        pl.xlim(self.auto['autosamples'].min(), self.auto['autosamples'].max())
        if (self.auto['autosamples']/self.auto['factor']).max() > self.auto['automax'].max():
            pl.ylim(self.auto['automin'].min(), (self.auto['autosamples']/self.auto['factor']).max())
        else:
            pl.ylim(self.auto['automin'].min(), self.auto['automax'].max())
        pl.xlabel('Steps per Parameter')
        pl.ylabel('Autocorrelation Time')
        pl.legend()

        fig.tight_layout()

        if self.saveplot is not None:
            fig.savefig(self.saveplot, dpi=150)
            print("Auto plot saved to %s" % self.saveplot)
        else:
            fig.show()


class CornerPlot(object):
    """
    Class to handle the creation of a corner plot from output
    MCMC chains and a posterior object.

    Args:
        post (radvel.Posterior): radvel posterior object
        chains (DataFrame): MCMC chains output by radvel.mcmc
        saveplot (str, optional):  Name of output file, will show as
            interactive matplotlib window if not defined.

    """
    def __init__(self, post, chains, saveplot=None):

        self.post = post
        self.chains = chains
        self.saveplot = saveplot

        self.labels = [k for k in post.params.keys() if post.params[k].vary]
        self.texlabels = [post.params.tex_labels().get(l, l) for l in self.labels]
        #Modified by CB 4/18/2022
        if 'gp_amp0' in post.params.keys():
            for i in range(len(self.texlabels)):
                if self.texlabels[i] == '$\\eta_{1}_{a}$':
                    self.texlabels[i] = '$\\eta_{0}$'
                else:
                    continue

    def plot(self):
        """
        Make and either save or display the corner plot
        """

        f = rcParams['font.size']
        rcParams['font.size'] = 12

        _ = corner.corner(
            self.chains[self.labels], labels=self.texlabels, label_kwargs={"fontsize": 14},
            plot_datapoints=False, bins=30, quantiles=[0.16, 0.5, 0.84],
            show_titles=True, title_kwargs={"fontsize": 14}, smooth=True
        )

        if self.saveplot is not None:
            pl.savefig(self.saveplot, dpi=150)
            print("Corner plot saved to %s" % self.saveplot)
        else:
            pl.show()

        rcParams['font.size'] = f


class DerivedPlot(object):
    """
    Class to handle the creation of a corner plot of derived parameters
    from output MCMC chains and a posterior object.

    Args:
        chains (DataFrame): MCMC chains output by radvel.mcmc
        P:  object representation of config file
        saveplot (Optional[string]: Name of output file, will show as
            interactive matplotlib window if not defined.

    """

    def __init__(self, chains, P, saveplot=None):

        self.chains = chains
        self.saveplot = saveplot

        if 'planet_letters' in dir(P):
            planet_letters = P.planet_letters
        else:
            planet_letters = {1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k'}

        # Determine which columns to include in corner plot
        self.labels = []
        self.texlabels = []
        self.units = []
        for i in np.arange(1, P.nplanets + 1, 1):
            letter = planet_letters[i]

            for key in 'mpsini rhop a'.split():
                label = '{}{}'.format(key, i)

                is_column = list(self.chains.columns).count(label) == 1
                if not is_column:
                    continue

                null_column = self.chains.isnull().any().loc[label]
                if null_column:
                    continue

                tl = texlabel(label, letter)

                # add units to label
                if key == 'mpsini':
                    unit = "M$_{\\oplus}$"
                    if np.median(self.chains[label]) > 100:
                        unit = "M$_{\\rm Jup}$"
                        self.chains[label] *= 0.00315
                    if np.median(self.chains[label]) > 100:
                        unit = "M$_{\\odot}$"
                        self.chains[label] *= 0.000954265748
                elif key == 'rhop':
                    unit = " g cm$^{-3}$"
                elif key == 'a':
                    unit = " AU"
                else:
                    unit = " "

                self.units.append(unit)
                self.labels.append(label)
                self.texlabels.append(tl)

    def plot(self):
        """
        Make and either save or display the corner plot
        """

        f = rcParams['font.size']
        rcParams['font.size'] = 12

        plot_labels = []
        for t, u in zip(self.texlabels, self.units):
            label = '{} [{}]'.format(t, u)
            plot_labels.append(label)

        _ = corner.corner(
            self.chains[self.labels], labels=plot_labels, label_kwargs={"fontsize": 14},
            plot_datapoints=False, bins=30, quantiles=[0.16, 0.50, 0.84],
            show_titles=True, title_kwargs={"fontsize": 14}, smooth=True
        )

        if self.saveplot is not None:
            pl.savefig(self.saveplot, dpi=150)
            print("Derived plot saved to %s" % self.saveplot)
        else:
            pl.show()

        rcParams['font.size'] = f


def texlabel(key, letter):
    """
    Args:
        key (list of string): list of parameter strings
        letter (string): planet letter

    Returns:
        string: LaTeX label for parameter string
    """
    if key.count('mpsini') == 1:
        return '$M_' + letter + '\\sin i$'
    if key.count('rhop') == 1:
        return '$\\rho_' + letter + '$'
    if key.count('a') == 1:
        return "$a_" + letter + "$"
