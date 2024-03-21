def linear_ld_law(mu, u1):
    """ Linear limb darkening law. """
    return 1. - u1 * (1. - mu)


def quadratic_ld_law(mu, u1, u2):
    """ Quadratic limb darkening law. """
    return 1. - u1 * (1. - mu) - u2 * (1. - mu)**2


def kipping_ld_law(mu, q1, q2):
    """ Kipping limb darkening law. """
    u1 = 2. * q1**0.5 * q2
    u2 = q1**0.5 * (1. - 2. * q2)
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
