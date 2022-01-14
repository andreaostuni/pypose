import torch
from torch import nn
from .broadcasting import broadcast_inputs
from .group_ops import exp, log, inv, mul, adj
from .group_ops import adjT, jinv, act3, act4, toMatrix, toVec, fromVec


class GroupType:
    '''Lie Group Type Base Class'''
    def __init__(self, groud,  dimension, embedding, manifold):
        self.group     = groud     # Group ID
        self.dimension = dimension # Data dimension
        self.embedding = embedding # Embedding dimension
        self.manifold  = manifold  # Manifold dimension

    def Log(self, x):
        raise NotImplementedError("Instance has no Log attribute.")
    
    def Exp(self, x):
        raise NotImplementedError("Instance has no Exp attribute.")

    def identity(cls, *args, **kwargs):
        raise NotImplementedError("Instance has no identity.")


class SO3Type(GroupType):
    def __init__(self):
        super().__init__(1, 4, 4, 3)

    def Log(self, x):
        inputs, out_shape = broadcast_inputs(x, None)
        out = log.apply(self.group, *inputs)
        return LieGroup(out.view(out_shape + (-1,)),
                gtype=so3_type, requires_grad=x.requires_grad)

    @classmethod
    def identity(cls, *args, **kwargs):
        data = torch.tensor([0., 0., 0., 1.], **kwargs)
        return LieGroup(data.expand(args+(-1,)),
                gtype=SO3_type, requires_grad=data.requires_grad)


class so3Type(GroupType):
    def __init__(self):
        super().__init__(1, 3, 4, 3)

    def Exp(self, x):
        inputs, out_shape = broadcast_inputs(x, None)
        out = exp.apply(self.group, *inputs)
        return LieGroup(out.view(out_shape + (-1,)),
                gtype=SO3_type, requires_grad=x.requires_grad)

    @classmethod
    def identity(cls, *args, **kwargs):
        return SO3_type.Log(SO3_type.identity(*args, **kwargs))


class SE3Type(GroupType):
    def __init__(self):
        super().__init__(3, 7, 7, 6)

    def Log(self, x):
        inputs, out_shape = broadcast_inputs(x, None)
        out = log.apply(self.group, *inputs)
        return LieGroup(out.view(out_shape + (-1,)),
                gtype=se3_type, requires_grad=x.requires_grad)

    @classmethod
    def identity(cls, *args, **kwargs):
        data = torch.tensor([0., 0., 0., 0., 0., 0., 1.], **kwargs)
        return LieGroup(data.expand(args+(-1,)),
                gtype=SE3_type, requires_grad=data.requires_grad)


class se3Type(GroupType):
    def __init__(self):
        super().__init__(3, 6, 7, 6)

    def Exp(self, x):
        inputs, out_shape = broadcast_inputs(x, None)
        out = exp.apply(self.group, *inputs)
        return LieGroup(out.view(out_shape + (-1,)),
                gtype=SO3_type, requires_grad=x.requires_grad)

    @classmethod
    def identity(cls, *args, **kwargs):
        return SE3_type.Log(SE3_type.identity(*args, **kwargs))


SO3_type, so3_type = SO3Type(), so3Type()
SE3_type, se3_type = SE3Type(), se3Type()


class LieGroup(torch.Tensor):
    """ Lie Group """
    from torch._C import _disabled_torch_function_impl
    __torch_function__ = _disabled_torch_function_impl

    def __init__(self, data, gtype=None, **kwargs):
        self.gtype = gtype

    def __new__(cls, data=None, **kwargs):
        return torch.Tensor.as_subclass(data, LieGroup) 

    def __repr__(self):
        return self.gtype.__class__.__name__ + " Group:\n" + super().__repr__()

    @property
    def gshape(self):
        return self.shape[:-1]
    
    def tensor(self):
        return self.data

    def Exp(self):
        return self.gtype.Exp(self)

    def Log(self):
        return self.gtype.Log(self)

#    def vec(self):
#        return self.apply_op(ToVec, self.data)

    @classmethod
    def IdentityLike(cls, G):
        return cls.Identity(G.shape, device=G.data.device, dtype=G.data.dtype)

    @classmethod
    def InitFromVec(cls, data):
        return cls(cls.apply_op(FromVec, data))

    @classmethod
    def apply_op(cls, op, x, y=None):
        """ Apply group operator """
        inputs, out_shape = broadcast_inputs(x, y)
        data = op.apply(cls.group_id, *inputs)
        return data.view(out_shape + (-1,))

    def Quaternion(self):
        """ extract quaternion """
        return self.apply_op(Quat, self.data)

    def Inv(self):
        """ group inverse """
        return self.__class__(self.apply_op(Inv, self.data))

    def Mul(self, other):
        """ group multiplication """
        return self.__class__(self.apply_op(Mul, self.data, other.data))

    def Retr(self, a):
        """ retraction: Exp(a) * X """
        dX = self.__class__.apply_op(Exp, a)
        return self.__class__(self.apply_op(Mul, dX, self.data))

    def Adj(self, a):
        """ adjoint operator: b = A(X) * a """
        return self.apply_op(Adj, self.data, a)

    def AdjT(self, a):
        """ transposed adjoint operator: b = a * A(X) """
        return self.apply_op(AdjT, self.data, a)

    def Jinv(self, a):
        return self.apply_op(Jinv, self.data, a)

    def Act(self, p):
        """ action on a point cloud """

        # action on point
        if p.shape[-1] == 3:
            return self.apply_op(Act3, self.data, p)

        # action on homogeneous point
        elif p.shape[-1] == 4:
            return self.apply_op(Act4, self.data, p)

    def matrix(self):
        """ convert element to 4x4 matrix """
        I = torch.eye(4, dtype=self.dtype, device=self.device)
        I = I.view([1] * (len(self.data.shape) - 1) + [4, 4])
        return self.__class__(self.data[...,None,:]).act(I).transpose(-1,-2)

    def translation(self):
        """ extract translation component """
        p = torch.as_tensor([0.0, 0.0, 0.0, 1.0], dtype=self.dtype, device=self.device)
        p = p.view([1] * (len(self.data.shape) - 1) + [4,])
        return self.apply_op(Act4, self.data, p)

#    def __mul__(self, other):
        # group multiplication
#        if isinstance(other, LieGroup):
#            return self.mul(other)

#        elif isinstance(other, torch.Tensor):
#            return self.act(other)


class Parameter(LieGroup, nn.Parameter):
    def __new__(cls, data=None, gtype=None, requires_grad=True):
        if data is None:
            data = torch.tensor([])
        return LieGroup._make_subclass(cls, data, requires_grad)
