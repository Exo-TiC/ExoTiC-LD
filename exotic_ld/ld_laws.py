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


def linear_ld_law(mu, u1):
    """ Linear limb darkening law. """
    return 1. - u1 * (1. - mu)


def quadratic_ld_law(mu, u1, u2):
    """ Quadratic limb darkening law. """
    return 1. - u1 * (1. - mu) - u2 * (1. - mu)**2


def squareroot_ld_law(mu, u1, u2):
    """ Square root limb darkening law. """
    return 1. - u1 * (1. - mu) - u2 * (1. - mu**0.5)


def nonlinear_3param_ld_law(mu, u1, u2, u3):
    """ Non-linear 3-parameter limb darkening law. """
    return 1. - u1 * (1. - mu) - u2 * (1. - mu**1.5) - u3 * (1. - mu**2)


def nonlinear_4param_ld_law(mu, u1, u2, u3, u4):
    """ Non-linear 4-parameter limb darkening law. """
    return 1. - u1 * (1. - mu**0.5) - u2 * (1. - mu) \
           - u3 * (1. - mu**1.5) - u4 * (1. - mu**2)
