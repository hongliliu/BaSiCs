import math
import numpy as np
from scipy import optimize
from warnings import filterwarnings, catch_warnings

from basics.utils import wrap_to_pi, in_ellipse, in_circle, in_array
from basics.masking_utils import fraction_in_mask


def _check_data_dim(data, dim):
    if data.ndim != 2 or data.shape[1] != dim:
        raise ValueError('Input data must have shape (N, %d).' % dim)


def _check_data_atleast_2D(data):
    if data.ndim < 2 or data.shape[1] < 2:
        raise ValueError('Input data must be at least 2D.')


def _norm_along_axis(x, axis):
    """
    NumPy < 1.8 does not support the `axis` argument for
    `np.linalg.norm`.
    """
    return np.sqrt(np.einsum('ij,ij->i', x, x))


class BaseModel(object):

    def __init__(self):
        self.params = None


class CircleModel(BaseModel):

    """Total least squares estimator for 2D circles.

    The functional model of the circle is::

        r**2 = (x - xc)**2 + (y - yc)**2

    This estimator minimizes the squared distances from all points to the
    circle::

        min{ sum((r - sqrt((x_i - xc)**2 + (y_i - yc)**2))**2) }

    A minimum number of 3 points is required to solve for the parameters.

    Attributes
    ----------
    params : tuple
        Circle model parameters in the following order `xc`, `yc`, `r`.

    """

    def estimate(self, data, params0=None):
        """Estimate circle model from data using total least squares.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """

        _check_data_dim(data, dim=2)

        x = data[:, 0]
        y = data[:, 1]
        # pre-allocate jacobian for all iterations
        A = np.zeros((3, data.shape[0]), dtype=np.double)
        # same for all iterations: r
        A[2, :] = -1

        def dist(xc, yc):
            return np.sqrt((x - xc)**2 + (y - yc)**2)

        def fun(params):
            xc, yc, r = params
            return dist(xc, yc) - r

        def Dfun(params):
            xc, yc, r = params
            d = dist(xc, yc)
            A[0, :] = -(x - xc) / d
            A[1, :] = -(y - yc) / d
            # same for all iterations, so not changed in each iteration
            #A[2, :] = -1
            return A

        if params0 is None:
            xc0 = np.median(x)
            yc0 = np.median(y)
            r0 = np.median(np.sqrt((x - xc0)**2 + (y - yc0)**2))
            params0 = (xc0, yc0, r0)
        else:
            if len(params0) != 3:
                raise ValueError("params0 must have a length of 3.")
            params0 = tuple(params0)

        pfit, pcov, infodict, errmsg, success = \
            optimize.leastsq(fun, params0, Dfun=Dfun, col_deriv=True,
                             full_output=True)

        self.params = pfit
        # Sometimes you get negative (but reasonable) radii out. Force to be
        # positive.
        self.params[-1] = np.abs(self.params[-1])

        # Need to multiply the fractional covariance matrix from leastsq with
        # the reduced chi-square value
        if len(data) > 3 and pcov is not None:
            s_sq = (self.residuals(data) ** 2).sum() / (len(data) - 3)
            pcov *= s_sq

            self.param_errors = np.empty((len(pfit)))
            # Standard errors are sqrt of the cov matrix diagonals
            for i in range(len(pfit)):
                self.param_errors[i] = np.sqrt(np.abs(pcov[i, i]))
        else:
            self.param_errors = np.array([np.NaN] * len(pfit))
        # Did it work?
        if success in [1, 2, 3, 4]:
            return True

        # If not, print fail message and return false
        # print(output[-2])

        return False

    def residuals(self, data):
        """Determine residuals of data to model.

        For each point the shortest distance to the circle is returned.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.

        """

        _check_data_dim(data, dim=2)

        xc, yc, r = self.params

        x = data[:, 0]
        y = data[:, 1]

        return np.abs(r - np.sqrt((x - xc)**2 + (y - yc)**2))

    def predict_xy(self, t, params=None):
        """Predict x- and y-coordinates using the estimated model.

        Parameters
        ----------
        t : array
            Angles in circle in radians. Angles start to count from positive
            x-axis to positive y-axis in a right-handed system.
        params : (3, ) array, optional
            Optional custom parameter set.

        Returns
        -------
        xy : (..., 2) array
            Predicted x- and y-coordinates.

        """
        if params is None:
            params = self.params
        xc, yc, r = params

        x = xc + r * np.cos(t)
        y = yc + r * np.sin(t)

        return np.concatenate((x[..., None], y[..., None]), axis=t.ndim)


class EllipseModel(BaseModel):

    """Total least squares estimator for 2D ellipses.

    The functional model of the ellipse is::

        xt = xc + a*cos(theta)*cos(t) - b*sin(theta)*sin(t)
        yt = yc + a*sin(theta)*cos(t) + b*cos(theta)*sin(t)
        d = sqrt((x - xt)**2 + (y - yt)**2)

    where ``(xt, yt)`` is the closest point on the ellipse to ``(x, y)``. Thus
    d is the shortest distance from the point to the ellipse.

    This estimator minimizes the squared distances from all points to the
    ellipse::

        min{ sum(d_i**2) } = min{ sum((x_i - xt)**2 + (y_i - yt)**2) }

    Thus you have ``2 * N`` equations (x_i, y_i) for ``N + 5`` unknowns (t_i,
    xc, yc, a, b, theta), which gives you an effective redundancy of ``N - 5``.

    The ``params`` attribute contains the parameters in the following order::

        xc, yc, a, b, theta

    A minimum number of 5 points is required to solve for the parameters.

    Attributes
    ----------
    params : tuple
        Ellipse model parameters in the following order `xc`, `yc`, `a`,
        `b`, `theta`.

    """

    def estimate(self, data, params0=None):
        """Estimate circle model from data using total least squares.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        success : bool
            True, if model estimation succeeds.

        """

        _check_data_dim(data, dim=2)

        x = data[:, 0]
        y = data[:, 1]

        N = data.shape[0]

        # pre-allocate jacobian for all iterations
        A = np.zeros((N + 5, 2 * N), dtype=np.double)
        # same for all iterations: xc, yc
        A[0, :N] = -1
        A[1, N:] = -1

        diag_idxs = np.diag_indices(N)

        def fun(params):
            xyt = self.predict_xy(params[5:], params[:5])
            fx = x - xyt[:, 0]
            fy = y - xyt[:, 1]
            return np.append(fx, fy)

        def Dfun(params):
            xc, yc, a, b, theta = params[:5]
            t = params[5:]

            ct = np.cos(t)
            st = np.sin(t)
            ctheta = math.cos(theta)
            stheta = math.sin(theta)

            # derivatives for fx, fy in the following order:
            #       xc, yc, a, b, theta, t_i

            # fx
            A[2, :N] = - ctheta * ct
            A[3, :N] = stheta * st
            A[4, :N] = a * stheta * ct + b * ctheta * st
            A[5:, :N][diag_idxs] = a * ctheta * st + b * stheta * ct
            # fy
            A[2, N:] = - stheta * ct
            A[3, N:] = - ctheta * st
            A[4, N:] = - a * ctheta * ct + b * stheta * st
            A[5:, N:][diag_idxs] = a * stheta * st - b * ctheta * ct

            return A

        # initial guess of parameters using a circle model
        all_params0 = np.empty((N + 5, ), dtype=np.double)
        if params0 is None:
            xc0 = np.median(x)
            yc0 = np.median(y)
            r0 = np.median(np.sqrt((x - xc0)**2 + (y - yc0)**2))
            params0 = (xc0, yc0, r0, r0, 0)
        else:
            if len(params0) != 5:
                raise ValueError("params0 must have a length of 5.")
            params0 = tuple(params0)
            xc0 = params0[0]
            yc0 = params0[1]

        all_params0[:5] = params0
        all_params0[5:] = np.arctan2(y - yc0, x - xc0)

        pfit, pcov, infodict, errmsg, success = \
            optimize.leastsq(fun, all_params0, Dfun=Dfun, col_deriv=True,
                             full_output=True)

        self.params = pfit[:5]
        self.params[2:4] = np.abs(self.params[2:4])
        self.params[-1] = wrap_to_pi(self.params[-1])

        # Need to multiply the fractional covariance matrix from leastsq with
        # the reduced chi-square value
        if len(data) > 5 and pcov is not None:
            s_sq = (self.residuals(data) ** 2).sum() / (len(data) - 5)
            pcov *= s_sq

            self.param_errors = np.empty((len(pfit)))
            # Standard errors are sqrt of the cov matrix diagonals
            for i in range(len(pfit)):
                self.param_errors[i] = np.sqrt(np.abs(pcov[i, i]))

        else:
            self.param_errors = np.array([np.NaN] * len(pfit))

        # Did it work?
        if success in [1, 2, 3, 4]:
            return True

        # If not, print fail message and return false
        # print(output[-2])

        return False

    def residuals(self, data):
        """Determine residuals of data to model.

        For each point the shortest distance to the ellipse is returned.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.

        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.

        """

        _check_data_dim(data, dim=2)

        xc, yc, a, b, theta = self.params

        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        x = data[:, 0]
        y = data[:, 1]

        N = data.shape[0]

        def fun(t, xi, yi):
            ct = math.cos(t)
            st = math.sin(t)
            xt = xc + a * ctheta * ct - b * stheta * st
            yt = yc + a * stheta * ct + b * ctheta * st
            return (xi - xt)**2 + (yi - yt)**2

        # def Dfun(t, xi, yi):
        #     ct = math.cos(t)
        #     st = math.sin(t)
        #     xt = xc + a * ctheta * ct - b * stheta * st
        #     yt = yc + a * stheta * ct + b * ctheta * st
        #     dfx_t = - 2 * (xi - xt) * (- a * ctheta * st
        #                                - b * stheta * ct)
        #     dfy_t = - 2 * (yi - yt) * (- a * stheta * st
        #                                + b * ctheta * ct)
        #     return [dfx_t + dfy_t]

        residuals = np.empty((N, ), dtype=np.double)

        # initial guess for parameter t of closest point on ellipse
        t0 = np.arctan2(y - yc, x - xc) - theta

        # determine shortest distance to ellipse for each point
        for i in range(N):
            xi = x[i]
            yi = y[i]
            # faster without Dfun, because of the python overhead
            t, _ = optimize.leastsq(fun, t0[i], args=(xi, yi))
            residuals[i] = np.sqrt(fun(t, xi, yi))

        return residuals

    def predict_xy(self, t, params=None):
        """Predict x- and y-coordinates using the estimated model.

        Parameters
        ----------
        t : array
            Angles in circle in radians. Angles start to count from positive
            x-axis to positive y-axis in a right-handed system.
        params : (5, ) array, optional
            Optional custom parameter set.

        Returns
        -------
        xy : (..., 2) array
            Predicted x- and y-coordinates.

        """

        if params is None:
            params = self.params
        xc, yc, a, b, theta = params

        ct = np.cos(t)
        st = np.sin(t)
        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        x = xc + a * ctheta * ct - b * stheta * st
        y = yc + a * stheta * ct + b * ctheta * st

        return np.concatenate((x[..., None], y[..., None]), axis=t.ndim)


def _dynamic_max_trials(n_inliers, n_samples, min_samples, probability):
    """Determine number trials such that at least one outlier-free subset is
    sampled for the given inlier/outlier ratio.
    Parameters
    ----------
    n_inliers : int
        Number of inliers in the data.
    n_samples : int
        Total number of samples in the data.
    min_samples : int
        Minimum number of samples chosen randomly from original data.
    probability : float
        Probability (confidence) that one outlier-free sample is generated.
    Returns
    -------
    trials : int
        Number of trials.
    """
    if n_inliers == 0:
        return np.inf

    nom = 1 - probability
    if nom == 0:
        return np.inf

    inlier_ratio = n_inliers / float(n_samples)
    denom = 1 - inlier_ratio ** min_samples
    if denom == 0:
        return 1
    elif denom == 1:
        return np.inf

    nom = np.log(nom)
    denom = np.log(denom)
    if denom == 0:
        return 0

    return int(np.ceil(nom / denom))


def ransac(data, model_class, min_samples, residual_threshold,
           is_data_valid=None, is_model_valid=None,
           max_trials=100, stop_sample_num=np.inf, stop_residuals_sum=0,
           stop_probability=1):
    """Fit a model to data with the RANSAC (random sample consensus) algorithm.

    RANSAC is an iterative algorithm for the robust estimation of parameters
    from a subset of inliers from the complete data set. Each iteration
    performs the following tasks:

    1. Select `min_samples` random samples from the original data and check
       whether the set of data is valid (see `is_data_valid`).
    2. Estimate a model to the random subset
       (`model_cls.estimate(*data[random_subset]`) and check whether the
       estimated model is valid (see `is_model_valid`).
    3. Classify all data as inliers or outliers by calculating the residuals
       to the estimated model (`model_cls.residuals(*data)`) - all data samples
       with residuals smaller than the `residual_threshold` are considered as
       inliers.
    4. Save estimated model as best model if number of inlier samples is
       maximal. In case the current estimated model has the same number of
       inliers, it is only considered as the best model if it has less sum of
       residuals.

    These steps are performed either a maximum number of times or until one of
    the special stop criteria are met. The final model is estimated using all
    inlier samples of the previously determined best model.

    Parameters
    ----------
    data : [list, tuple of] (N, D) array
        Data set to which the model is fitted, where N is the number of data
        points and D the dimensionality of the data.
        If the model class requires multiple input data arrays (e.g. source and
        destination coordinates of  ``skimage.transform.AffineTransform``),
        they can be optionally passed as tuple or list. Note, that in this case
        the functions ``estimate(*data)``, ``residuals(*data)``,
        ``is_model_valid(model, *random_data)`` and
        ``is_data_valid(*random_data)`` must all take each data array as
        separate arguments.
    model_class : object
        Object with the following object methods:

         * ``success = estimate(*data)``
         * ``residuals(*data)``

        where `success` indicates whether the model estimation succeeded
        (`True` or `None` for success, `False` for failure).
    min_samples : int
        The minimum number of data points to fit a model to.
    residual_threshold : float
        Maximum distance for a data point to be classified as an inlier.
    is_data_valid : function, optional
        This function is called with the randomly selected data before the
        model is fitted to it: `is_data_valid(*random_data)`.
    is_model_valid : function, optional
        This function is called with the estimated model and the randomly
        selected data: `is_model_valid(model, *random_data)`, .
    max_trials : int, optional
        Maximum number of iterations for random sample selection.
    stop_sample_num : int, optional
        Stop iteration if at least this number of inliers are found.
    stop_residuals_sum : float, optional
        Stop iteration if sum of residuals is less than or equal to this
        threshold.
    stop_probability : float in range [0, 1], optional
        RANSAC iteration stops if at least one outlier-free set of the
        training data is sampled with ``probability >= stop_probability``,
        depending on the current best model's inlier ratio and the number
        of trials. This requires to generate at least N samples (trials):

            N >= log(1 - probability) / log(1 - e**m)

        where the probability (confidence) is typically set to a high value
        such as 0.99, and e is the current fraction of inliers w.r.t. the
        total number of samples.

    Returns
    -------
    model : object
        Best model with largest consensus set.
    inliers : (N, ) array
        Boolean mask of inliers classified as ``True``.

    References
    ----------
    .. [1] "RANSAC", Wikipedia, http://en.wikipedia.org/wiki/RANSAC

    Examples
    --------

    Generate ellipse data without tilt and add noise:

    >>> t = np.linspace(0, 2 * np.pi, 50)
    >>> a = 5
    >>> b = 10
    >>> xc = 20
    >>> yc = 30
    >>> x = xc + a * np.cos(t)
    >>> y = yc + b * np.sin(t)
    >>> data = np.column_stack([x, y])
    >>> np.random.seed(seed=1234)
    >>> data += np.random.normal(size=data.shape)

    Add some faulty data:

    >>> data[0] = (100, 100)
    >>> data[1] = (110, 120)
    >>> data[2] = (120, 130)
    >>> data[3] = (140, 130)

    Estimate ellipse model using all available data:

    >>> model = EllipseModel()
    >>> model.estimate(data)
    True
    >>> model.params # doctest: +SKIP
    array([ -3.30354146e+03,  -2.87791160e+03,   5.59062118e+03,
             7.84365066e+00,   7.19203152e-01])


    Estimate ellipse model using RANSAC:

    >>> ransac_model, inliers = ransac(data, EllipseModel, 5, 3, max_trials=50)
    >>> ransac_model.params
    array([ 20.12762373,  29.73563063,   4.81499637,  10.4743584 ,   0.05217117])
    >>> inliers
    array([False, False, False, False,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True], dtype=bool)

    Robustly estimate geometric transformation:

    >>> from skimage.transform import SimilarityTransform
    >>> np.random.seed(0)
    >>> src = 100 * np.random.rand(50, 2)
    >>> model0 = SimilarityTransform(scale=0.5, rotation=1,
    ...                              translation=(10, 20))
    >>> dst = model0(src)
    >>> dst[0] = (10000, 10000)
    >>> dst[1] = (-100, 100)
    >>> dst[2] = (50, 50)
    >>> model, inliers = ransac((src, dst), SimilarityTransform, 2, 10)
    >>> inliers
    array([False, False, False,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True], dtype=bool)

    """

    best_model = None
    best_inlier_num = 0
    best_inlier_residuals_sum = np.inf
    best_inliers = None

    if min_samples < 0:
        raise ValueError("`min_samples` must be greater than zero")

    if max_trials < 0:
        raise ValueError("`max_trials` must be greater than zero")

    if stop_probability < 0 or stop_probability > 1:
        raise ValueError("`stop_probability` must be in range [0, 1]")

    if not isinstance(data, list) and not isinstance(data, tuple):
        data = [data]

    # make sure data is list and not tuple, so it can be modified below
    data = list(data)
    # number of samples
    num_samples = data[0].shape[0]

    for num_trials in range(max_trials):

        # choose random sample set
        samples = []
        random_idxs = np.random.choice(num_samples, min_samples,
                                       replace=False)
        for d in data:
            samples.append(d[random_idxs])

        # check if random sample set is valid
        if is_data_valid is not None and not is_data_valid(*samples):
            continue

        # estimate model for current random sample set
        sample_model = model_class()

        success = sample_model.estimate(*samples)

        if success is not None:  # backwards compatibility
            if not success:
                continue

        # check if estimated model is valid
        if is_model_valid is not None and not is_model_valid(sample_model,
                                                             *samples):
            continue

        sample_model_residuals = np.abs(sample_model.residuals(*data))
        # consensus set / inliers
        sample_model_inliers = sample_model_residuals < residual_threshold
        sample_model_residuals_sum = np.sum(sample_model_residuals**2)

        # choose as new best model if number of inliers is maximal
        sample_inlier_num = np.sum(sample_model_inliers)
        if (
            # more inliers
            sample_inlier_num > best_inlier_num
            # same number of inliers but less "error" in terms of residuals
            or (sample_inlier_num == best_inlier_num
                and sample_model_residuals_sum < best_inlier_residuals_sum)
        ):
            best_model = sample_model
            best_inlier_num = sample_inlier_num
            best_inlier_residuals_sum = sample_model_residuals_sum
            best_inliers = sample_model_inliers
            if (
                best_inlier_num >= stop_sample_num
                or best_inlier_residuals_sum <= stop_residuals_sum
                or num_trials
                    >= _dynamic_max_trials(best_inlier_num, num_samples,
                                           min_samples, stop_probability)
            ):
                break

    # estimate final model using all inliers
    if best_inliers is not None:
        # select inliers for each data array
        for i in range(len(data)):
            data[i] = data[i][best_inliers]
        best_model.estimate(*data)

    return best_model, best_inliers


def fit_region(coords, initial_props=None,
               try_fit_ellipse=True, use_ransac=False,
               min_in_mask=0.8, mask=None,
               ransac_trials=50, beam_pix=4, max_rad=1.75,
               max_eccent=3., image_shape=None, verbose=False):
    '''
    Fit a circle or ellipse to the given coordinates.
    '''

    coords = np.array(coords).copy()
    ymean = coords[:, 0].mean()
    xmean = coords[:, 1].mean()
    # Fitting works better when the points are near the origin
    coords[:, 0] -= int(ymean)
    coords[:, 1] -= int(xmean)
    new_props = np.empty((5,))

    if len(coords) < 3:
        raise ValueError("Must have at least 3 points to fit.")

    if len(coords) > 5 and try_fit_ellipse:
        with catch_warnings():
            filterwarnings("ignore",
                           r"Number of calls to function")
            filterwarnings("ignore",
                           r"gtol=0.000000 is too small")
            if use_ransac:
                model, inliers = \
                    ransac(coords[:, ::-1], EllipseModel, 5,
                           beam_pix, max_trials=ransac_trials)
            else:
                model = EllipseModel()
                model.estimate(coords[:, ::-1])

        dof = len(coords) - 5

        pars = model.params.copy()
        pars[0] += int(xmean)
        pars[1] += int(ymean)

        eccent = pars[2] / float(pars[3])
        # Sometimes a < b?? If so, manually correct.
        if eccent < 1:
            eccent = 1. / eccent
            pars[2], pars[3] = pars[3], pars[2]
            pars[4] = wrap_to_pi(pars[4] + 0.5 * np.pi)

        fail_conds = eccent > 3.

        if beam_pix is not None:
            fail_conds = fail_conds or \
                pars[3] < beam_pix

        if initial_props is not None:
            fail_conds = fail_conds or \
                pars[2] > max_rad * initial_props[2] or \
                not in_ellipse(initial_props[:2][::-1], pars)

        if image_shape is not None:
            fail_conds = fail_conds or \
                not in_array(pars[:2], image_shape)

        if mask is not None:
            fail_conds = fail_conds or \
                fraction_in_mask(pars, mask) < min_in_mask

        if fail_conds:
            ellip_fail = True
        else:
            new_props[0] = pars[1]
            new_props[1] = pars[0]
            new_props[2] = pars[2]
            new_props[3] = pars[3]
            new_props[4] = pars[4]
            ellip_fail = False
    else:
        ellip_fail = True

    # If ellipse fitting is not allowed, or it failed, fit a circle
    if ellip_fail:
        with catch_warnings():
            filterwarnings("ignore",
                           r"Number of calls to function")
            filterwarnings("ignore",
                           r"gtol=0.000000 is too small")
            if use_ransac:
                model, inliers = \
                    ransac(coords[:, ::-1], CircleModel, 3,
                           beam_pix, max_trials=ransac_trials)
            else:
                model = CircleModel()
                model.estimate(coords[:, ::-1])

        dof = len(coords) - 3

        pars = model.params.copy()
        pars[0] += int(xmean)
        pars[1] += int(ymean)

        fail_conds = False

        if beam_pix is not None:
            fail_conds = fail_conds or \
                pars[2] < beam_pix

        if initial_props is not None:
            fail_conds = fail_conds or \
                pars[2] > max_rad * initial_props[2] or \
                not in_circle(initial_props[:2][::-1], pars)

        if image_shape is not None:
            fail_conds = fail_conds or \
                not in_array(pars[:2], image_shape)

        if mask is not None:
            fail_conds = fail_conds or \
                fraction_in_mask(pars, mask) < min_in_mask

        if fail_conds:
            if verbose:
                Warning("All fitting failed.")
            return None, None

        new_props[0] = pars[1]
        new_props[1] = pars[0]
        new_props[2] = pars[2]
        new_props[3] = pars[2]
        new_props[4] = 0.0

    props = new_props

    # Calculate the model residual
    resid = model.residuals(coords[:, ::-1]).sum() / dof

    # coords[:, 0] += int(ymean)
    # coords[:, 1] += int(xmean)

    return props, resid
