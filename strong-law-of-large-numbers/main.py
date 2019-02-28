import numpy as np
from scipy.stats import cauchy, norm, pareto, uniform
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

sns.set_context("poster")
sns.set_style("whitegrid")


def generate_from(distribution, *args):
    """
    Returns a generator object that yields a new random variable according to
    the specified distribution, together with the generator's time averages up
    til the current step.
    """
    iterations = 0
    realization = distribution(*args)
    time_average = 0.
    while True:
        sample = realization.rvs()
        time_average = (iterations * time_average + sample) / (iterations + 1)
        iterations += 1
        yield sample, time_average


class DistributionClass():
    """
    TODO: rename

    This class performs all of the random variable generation logic for the
    purpose of animating it in the other class.
    """

    def __init__(self, distribution, *args):
        self.dist = distribution
        self.dist_args = args

    def init_rvs(self):
        self.rv_generator = generate_from(self.dist, *self.dist_args)
        rv, mean = next(self.rv_generator)
        self.sample_rvs = np.array([rv], ndmin=2)
        self.sample_means = np.array([mean], ndmin=2)
        self.count = 1

        return self.sample_rvs, self.sample_means

    def next_rvs(self, i):
        rv, mean = next(self.rv_generator)
        self.sample_rvs = np.append(self.sample_rvs, [rv], axis=0)
        self.sample_means = np.append(self.sample_means, [mean], axis=0)
        self.count += 1
        
        return self.sample_rvs, self.sample_means


class AnimatedScatterPlot():
    """
    """

    def __init__(self, distribution, *args, xlim=(-1.,1.), ylim=(-1.,1.)):
        self.dist_thing = DistributionClass(distribution, *args)
        self.default_filename = 'animations/' +\
                f'{self.dist_thing.dist.name}_' +\
                '_'.join(str(arg) for arg in self.dist_thing.dist_args) +\
                '.mp4'

        self.fig, self.axes = plt.subplots(
                1, 2, sharex=True, sharey=True, 
                subplot_kw={ 
                    'aspect': 'equal', 
                    'autoscale_on': False, 
                    'xlim': xlim, 
                    'ylim': ylim 
                    }, 
                figsize=(15,8)
                )

        self.colors = lambda count, rev=False: \
                sns.color_palette("Reds" + ("_r" if rev else ""), count)

        sns.despine(self.fig, left=True, bottom=True)
        self.fig.suptitle(
                "IID random variables disributed like " +\
                        f"{self.dist_thing.dist.name}(" +\
                        ','.join(str(arg) for arg in self.dist_thing.dist_args)
                        + ")"
                )

        self.animation = animation.FuncAnimation(self.fig, 
                self.update_animation, 
                frames=1000,
                interval=50,
                init_func=self.init_animation,
                blit=True)

    def init_animation(self):
        local_kwargs = {
                'animated': True,
                'marker': '.',
                'c': self.colors(1, rev=True)
                }

        rvs, means = self.dist_thing.init_rvs()

        self.sample_scatter = self.axes[0].scatter(
                rvs[0, 0], rvs[0, 1], 
                **local_kwargs)
        self.mean_scatter = self.axes[1].scatter(
                means[0, 0], rvs[0, 1], 
                **local_kwargs)

        # Titles
        self.axes[0].set_title("Samples ($X_1, X_2, X_3, \ldots$)")
        self.axes[1].set_title(
                "Sample means ($n^{-1}\sum_{k=1}^{n} X_k$, $n = 1, 2, \ldots$)"
                )

        return self.sample_scatter, self.mean_scatter,

    def update_animation(self, i):
        rvs, means = self.dist_thing.next_rvs(i)

        for scatter, data in zip([self.sample_scatter, self.mean_scatter], [rvs, means]):
            scatter.set_offsets(data)
            scatter.set_color(self.colors(self.dist_thing.count))

        return self.sample_scatter, self.mean_scatter,

    def save(self, filename=None):
        filename = filename or self.default_filename
        self.animation.save(filename, extra_args=['-vcodec', 'libx264'])


if __name__ == '__main__':
    distribution = pareto 
    args = [(0.5, 0.5)]
    demo = AnimatedScatterPlot(
            distribution, *args,
            xlim=(1.,500.), 
            ylim=(1.,500.)
            )
    demo.save()
