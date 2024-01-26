"""Hyperbolic operations utils functions."""

import torch

MAX_NORM = 1e6
MIN_NORM = 1e-15
BALL_EPS = {torch.float32: 4e-3, torch.float64: 1e-5}


# ################# MATH FUNCTIONS ########################

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        dtype = x.dtype
        x = x.double()
        return (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def cosh(x, clamp=15):
    return x.clamp(-clamp, clamp).cosh()


def sinh(x, clamp=15):
    return x.clamp(-clamp, clamp).sinh()


def tanh(x):
    return x.clamp(-15, 15).tanh()


class Arcosh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(min=1.0 + 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (z + torch.sqrt_(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (input ** 2 - 1) ** 0.5


def arcosh(x):
    return Arcosh.apply(x)


# ################# HYPERBOLIC FUNCTION WITH CURVATURES=1 ########################


def givens_rotations(r, x):
    """Givens rotations.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    # [batch_size, dim // 2, 2]
    givens = r.view((r.shape[0], -1, 2))
    # L2 norm to constrain sine/cosine matrix
    givens = givens / torch.norm(givens, p=2, dim=-1,
                                 keepdim=True).clamp_min(1e-15)
    # [batch_size, dim // 2, 2]
    x = x.view((r.shape[0], -1, 2))
    # cosine: givens[:, :, 0:1] (batch_size, dim//2, 1)
    # sine: givens[:, :, 1:]  (batch_size, dim//2, 1)
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * \
            torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))


def givens_reflection(r, x):
    """Givens reflections.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to reflect

    Returns:
        torch.Tensor os shape (N x d) representing reflection of x by r
    """
    # [batch_size, dim // 2, 2]
    givens = r.view((r.shape[0], -1, 2))
    # L2 norm to constrain sine/cosine matrix
    givens = givens / torch.norm(givens, p=2, dim=-1,
                                 keepdim=True).clamp_min(1e-15)
    # [batch_size, dim // 2, 2]
    x = x.view((r.shape[0], -1, 2))
    # cosine: givens[:, :, 0:1] (batch_size, dim//2, 1)
    # sine: givens[:, :, 1:]  (batch_size, dim//2, 1)
    x_ref = givens[:, :, 0:1] * torch.cat((x[:, :, 0:1], -x[:, :, 1:]), dim=-1) + givens[:, :, 1:] * torch.cat(
        (x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_ref.view((r.shape[0], -1))


# ################# constant curvature=1 ########################


def _lambda_x(x, c=1):
    x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
    return 2 / (1. - c * x_sqnorm).clamp_min(MIN_NORM)


def expmap(u, base):
    """Exponential map u in the tangent space of point base with curvature c.
        from NIPS18 Hyperbolic Neural Networks

    Args:
        u: torch.Tensor of size B x d with tangent points
        base: torch.Tensor of size B x d with hyperbolic points

    Returns:
        torch.Tensor with  hyperbolic points.
    """
    # p is in hyperbolic space
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    second_term = (
            tanh(1 / 2 * _lambda_x(base) * u_norm)
            * u
            / (u_norm)
    )
    gamma_1 = mobius_add(base, second_term)
    return gamma_1


def expmap0(u):
    """Exponential map taken at the origin of the Poincare ball with curvature c.

    Args:
        u: torch.Tensor of size B x d with tangent points

    Returns:
        torch.Tensor with  hyperbolic points.
    """
    # see equation 1 for detail
    u_norm = u.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    gamma_1 = tanh(u_norm) * u / (u_norm)
    return project(gamma_1)


def logmap(y, base):
    """Logarithmic map taken at the x of the Poincare ball with curvature c.
       from NIPS18 Hyperbolic Neural Networks

    Args:
        y: torch.Tensor of size B x d with hyperbolic points
        base: torch.Tensor of size B x d with hyperbolic points

    Returns:
        torch.Tensor with tangent points.
    """

    sub = mobius_add(-base, y)
    sub_norm = sub.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    lam = _lambda_x(base)
    return 2 / lam * artanh(sub_norm) * sub / sub_norm


def logmap0(y):
    """Logarithmic map taken at the origin of the Poincare ball with curvature c.

    Args:
        y: torch.Tensor of size B x d with hyperbolic points

    Returns:
        torch.Tensor with tangent points.
    """
    y_norm = y.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    return y / y_norm / artanh(y_norm)


def project(x):
    """Project points to Poincare ball with curvature c.
    Need to make sure hyperbolic embeddings are inside the unit ball.

    Args:
        x: torch.Tensor of size B x d with hyperbolic points
    Returns:
        torch.Tensor with projected hyperbolic points.
    """
    # [batch_size, dim]
    norm = x.norm(dim=-1, p=2, keepdim=True).clamp_min(MIN_NORM)
    eps = BALL_EPS[x.dtype]
    maxnorm = (1 - eps)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def mobius_add(x, y):
    """Mobius addition of points in the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic points
        y: torch.Tensor of size B x d with hyperbolic points

    Returns:
        Tensor of shape B x d representing the element-wise Mobius addition of x and y.
    """
    x2 = torch.sum(x * x, dim=-1, keepdim=True)
    y2 = torch.sum(y * y, dim=-1, keepdim=True)
    xy = torch.sum(x * y, dim=-1, keepdim=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    denom = 1 + 2 * xy + x2 * y2
    return num / denom.clamp_min(MIN_NORM)


def hyp_distance(x, y):
    """Hyperbolic distance on the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)

    Returns: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    dist_c = artanh(mobius_add(-x, y).norm(dim=-1, p=2, keepdim=False))
    dist = dist_c * 2
    return dist


def sq_hyp_distance(x, y):
    """Square Hyperbolic distance on the Poincare ball with curvature c.

    Args:
        x: torch.Tensor of size B x d with hyperbolic queries
        y: torch.Tensor with hyperbolic queries, shape n_entities x d if eval_mode is true else (B x d)

    Returns: torch,Tensor with hyperbolic distances, size B x 1 if eval_mode is False
            else B x n_entities matrix with all pairs distances
    """
    dist_c = artanh(mobius_add(-x, y).norm(dim=-1, p=2, keepdim=False))
    dist = dist_c * 2
    return dist ** 2


def poincare_to_lor(x):
    sqnorm = torch.norm(x, p=2, dim=1, keepdim=True) ** 2
    return torch.cat([1 + sqnorm, 2 * x], dim=1) / (1 - sqnorm)


def _gyration(u, v, w, dim: int = -1):
    u2 = u.pow(2).sum(dim=dim, keepdim=True)
    v2 = v.pow(2).sum(dim=dim, keepdim=True)
    uv = (u * v).sum(dim=dim, keepdim=True)
    uw = (u * w).sum(dim=dim, keepdim=True)
    vw = (v * w).sum(dim=dim, keepdim=True)
    a = -uw * v2 + vw + 2 * uv * vw
    b = -vw * u2 - uw
    d = 1 + 2 * 1 * uv + u2 * v2
    return w + 2 * (a * u + b * v) / d.clamp_min(MIN_NORM)


def ptransp(x, y, u):
    lambda_x = _lambda_x(x)
    lambda_y = _lambda_x(y)
    return _gyration(y, -x, u) * lambda_x / lambda_y


def ptransp0(x, u):
    lambda_x = _lambda_x(x)
    return 2 * u / lambda_x.clamp_min(MIN_NORM)


# ########################### lorentz part ###########################


def minkowski_dot(x, y, keepdim=True):
    res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
    if keepdim:
        res = res.view(res.shape + (1,))
    return res


def minkowski_norm(u, keepdim=True):
    dot = minkowski_dot(u, u, keepdim=keepdim)
    return torch.sqrt(torch.clamp(dot, min=BALL_EPS[u.dtype]))


def lor_distance(x, y):
    prod = minkowski_dot(x, y)
    theta = torch.clamp(-prod, min=1.0 + BALL_EPS[x.dtype])
    lordist = arcosh(theta) ** 2
    # clamp distance to avoid nans in Fermi-Dirac decoder
    return torch.clamp(lordist, max=50.0)


def lor_proj(x):
    d = x.size(-1) - 1
    y = x.narrow(-1, 1, d)
    y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
    mask = torch.ones_like(x)
    mask[:, 0] = 0
    vals = torch.zeros_like(x)
    vals[:, 0:1] = torch.sqrt(torch.clamp(1 + y_sqnorm, min=BALL_EPS[x.dtype]))
    return vals + mask * x


def lor_proj_tan(u, x):
    d = x.size(1) - 1
    ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
    mask = torch.ones_like(u)
    mask[:, 0] = 0
    vals = torch.zeros_like(u)
    vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=BALL_EPS[x.dtype])
    return vals + mask * u


def lor_proj_tan0(u):
    narrowed = u.narrow(-1, 0, 1)
    vals = torch.zeros_like(u)
    vals[:, 0:1] = narrowed
    return u - vals


def lor_expmap(u, x):
    normu = minkowski_norm(u)
    normu = torch.clamp(normu, max=MAX_NORM)
    theta = normu
    theta = torch.clamp(theta, min=MIN_NORM)
    result = cosh(theta) * x + sinh(theta) * u / theta
    return lor_proj(result)


def lor_logmap(x, y):
    xy = torch.clamp(minkowski_dot(x, y) + 1, max=-BALL_EPS[x.dtype]) - 1
    u = y + xy * x
    normu = minkowski_norm(u)
    normu = torch.clamp(normu, min=MIN_NORM)
    dist = lor_distance(x, y) ** 0.5
    result = dist * u / normu
    return lor_proj_tan(result, x)


def lor_expmap0(u):
    d = u.size(-1) - 1
    x = u.narrow(-1, 1, d).view(-1, d)
    x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x_norm = torch.clamp(x_norm, min=MIN_NORM)
    theta = x_norm
    res = torch.ones_like(u)
    res[:, 0:1] = cosh(theta)
    res[:, 1:] = sinh(theta) * x / x_norm
    return lor_proj(res)


def lor_logmap0(x):
    d = x.size(-1) - 1
    y = x.narrow(-1, 1, d).view(-1, d)
    y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
    y_norm = torch.clamp(y_norm, min=MIN_NORM)
    res = torch.zeros_like(x)
    theta = torch.clamp(x[:, 0:1], min=1.0 + BALL_EPS[x.dtype])
    res[:, 1:] = arcosh(theta) * y / y_norm
    return res


def lor_mobius_add(x, y):
    u = lor_logmap0(y)
    v = lor_ptransp0(x, u)
    return lor_expmap(v, x)


def lor_mobius_matvec(m, x):
    u = lor_logmap0(x)
    mu = u @ m.transpose(-1, -2)
    return lor_expmap0(mu)


def lor_ptransp(x, y, u):
    logxy = lor_logmap(x, y)
    logyx = lor_logmap(y, x)
    sqdist = torch.clamp(lor_distance(x, y), min=MIN_NORM)
    alpha = minkowski_dot(logxy, u) / sqdist
    res = u - alpha * (logxy + logyx)
    return lor_proj_tan(res, y)


def lor_ptransp0(x, u):
    x0 = x.narrow(-1, 0, 1)
    d = x.size(-1) - 1
    y = x.narrow(-1, 1, d)
    y_norm = torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=MIN_NORM)
    y_normalized = y / y_norm
    v = torch.ones_like(x)
    v[:, 0:1] = - y_norm
    v[:, 1:] = (1 - x0) * y_normalized
    alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True)
    res = u - alpha * v
    return lor_proj_tan(res, x)


def lor_to_poincare(x):
    d = x.size(-1) - 1
    return x.narrow(-1, 1, d) / (x[:, 0:1] + 1)
