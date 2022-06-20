from astropy.modeling.models import custom_model


@custom_model
def quadratic_limb_darkening(mu, a_ld=0., b_ld=0.):
    """ Define quadratic limb darkening model with two params. """
    return 1. - a_ld * (1. - mu) - b_ld * (1. - mu)**2


@custom_model
def nonlinear_limb_darkening(mu, c0=0., c1=0., c2=0., c3=0.):
    """ Define non-linear limb darkening model with four params. """
    return (1. - (c0 * (1. - mu**0.5) + c1 * (1. - mu)
            + c2 * (1. - mu**1.5) + c3 * (1. - mu**2)))
