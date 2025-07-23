def linear_ld_law(mu, u1):
    """ Linear limb-darkening law.

    .. math::

                \\frac{I(\\mu)}{I(\\mu = 1)} = 1 - u_1 (1 - \\mu),

    where :math:`\\mu = \\sqrt{1 - r^2}`, and :math:`u_1` is the
    limb-darkening coefficient.

    """
    return 1. - u1 * (1. - mu)


def quadratic_ld_law(mu, u1, u2):
    """ Quadratic limb-darkening law.

    .. math::

                \\frac{I(\\mu)}{I(\\mu = 1)} = 1 - u_1 (1 - \\mu) - u_2 (1 - \\mu)^2,

    where :math:`\\mu = \\sqrt{1 - r^2}`, and :math:`u_1`, :math:`u_2`
    are the limb-darkening coefficients.

    """
    return 1. - u1 * (1. - mu) - u2 * (1. - mu)**2


def kipping_ld_law(mu, q1, q2):
    """ Kipping limb-darkening law.

    .. math::

                \\frac{I(\\mu)}{I(\\mu = 1)} = 1 - u_1 (1 - \\mu) - u_2 (1 - \\mu)^2,

    where,

        .. math::

                u_1 &= 2 \\sqrt{q_1} q_2,

                u_2 &= \\sqrt{q_1} (1 - 2 q_2),

    and :math:`\\mu = \\sqrt{1 - r^2}`, and :math:`q_1`, :math:`q_2`
    are the limb-darkening coefficients.

    """
    u1 = 2. * q1**0.5 * q2
    u2 = q1**0.5 * (1. - 2. * q2)
    return 1. - u1 * (1. - mu) - u2 * (1. - mu)**2


def squareroot_ld_law(mu, u1, u2):
    """ Square root limb-darkening law.

    .. math::

                \\frac{I(\\mu)}{I(\\mu = 1)} = 1 - u_1 (1 - \\mu) - u_2 (1 - \\sqrt{\\mu}),

    where :math:`\\mu = \\sqrt{1 - r^2}`, and :math:`u_1`, :math:`u_2`
    are the limb-darkening coefficients.

    """
    return 1. - u1 * (1. - mu) - u2 * (1. - mu**0.5)


def nonlinear_3param_ld_law(mu, u1, u2, u3):
    """ Non-linear 3-parameter limb-darkening law.

    .. math::

                \\frac{I(\\mu)}{I(\\mu = 1)} = 1 - u_1 (1 - \\mu) - u_2 (1 - \\mu^{1.5}) - u_3 (1 - \\mu^2),

    where :math:`\\mu = \\sqrt{1 - r^2}`, and :math:`u_1`, :math:`u_2`,
    :math:`u_3` are the limb-darkening coefficients.

    """
    return 1. - u1 * (1. - mu) - u2 * (1. - mu**1.5) - u3 * (1. - mu**2)


def nonlinear_4param_ld_law(mu, u1, u2, u3, u4):
    """ Non-linear 4-parameter limb-darkening law.

    .. math::

                \\frac{I(\\mu)}{I(\\mu = 1)} = 1 - u_1 (1 - \\mu^{0.5}) - u_2 (1 - \\mu) - u_2 (1 - \\mu^{1.5}) - u_3 (1 - \\mu^2),

    where :math:`\\mu = \\sqrt{1 - r^2}`, and :math:`u_1`, :math:`u_2`,
    :math:`u_3`, :math:`u_4` are the limb-darkening coefficients.

    """
    return 1. - u1 * (1. - mu**0.5) - u2 * (1. - mu) \
           - u3 * (1. - mu**1.5) - u4 * (1. - mu**2)

def power2_ld_law(mu, c, alpha):
    """ Power-2 limb darkening law. """
    return 1. - c * (1. - mu**alpha)

