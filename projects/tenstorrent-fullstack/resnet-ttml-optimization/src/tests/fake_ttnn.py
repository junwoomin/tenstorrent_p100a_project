from types import SimpleNamespace
import torch
import torch.nn.functional as F


class Tensor:
    def __init__(self,value,dtype):
        self.value=value
        self.dtype=dtype
    @property
    def shape(self):
        return self.value.shape


class Config:
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)


class FakeTTNN:
    bfloat16='bfloat16'
    bfloat8_b='bfloat8_b'
    float32='float32'
    ROW_MAJOR_LAYOUT='row'
    TILE_LAYOUT='tile'
    DRAM_MEMORY_CONFIG='dram'
    L1_MEMORY_CONFIG='l1'
    TensorMemoryLayout=SimpleNamespace(HEIGHT_SHARDED='height',BLOCK_SHARDED='block',WIDTH_SHARDED='width')
    UnaryOpType=SimpleNamespace(RELU='relu')
    Conv2dConfig=Config
    def __init__(self):
        self.uploads=0
        self.fail_upload=None
        self.conv_calls=[]
    def UnaryWithParam(self,op):
        return op
    def from_torch(self,x,dtype='bfloat16',**options):
        self.uploads+=1
        if self.uploads == self.fail_upload:
            raise RuntimeError('Injected upload failure')
        dtype_torch=torch.float32 if dtype=='float32' else torch.bfloat16
        return Tensor(x.detach().to(dtype_torch).clone(),dtype)
    def to_torch(self,x):
        return x.value.clone()
    def from_device(self,x):
        return x
    def to_layout(self,x,layout):
        return x
    def to_memory_config(self,x,config):
        return x
    def reshape(self,x,shape):
        return Tensor(x.value.reshape(shape),x.dtype)
    def typecast(self,x,dtype):
        return Tensor(x.value.to(torch.float32 if dtype=='float32' else torch.bfloat16),dtype)
    def synchronize_device(self,device):
        pass
    def conv2d(self,**k):
        self.conv_calls.append(k)
        n,h,w,c=k['batch_size'],k['input_height'],k['input_width'],k['in_channels']
        x=k['input_tensor'].value.reshape(n,h,w,c).permute(0,3,1,2).float()
        bias=k['bias_tensor']
        y=F.conv2d(x,k['weight_tensor'].value.float(),None if bias is None else bias.value.flatten().float(),
                   k['stride'],k['padding'],k['dilation'],k['groups'])
        if k['conv_config'].activation == 'relu':
            y=F.relu(y)
        oh,ow=y.shape[2:]
        y=self.typecast(Tensor(y.permute(0,2,3,1).reshape(1,1,n*oh*ow,-1),k['dtype']),k['dtype'])
        return (y,(oh,ow),(k['weight_tensor'],bias)) if k['return_weights_and_bias'] else (y,(oh,ow))
    def max_pool2d(self,**k):
        n,h,w,c=k['batch_size'],k['input_h'],k['input_w'],k['channels']
        x=k['input_tensor'].value.reshape(n,h,w,c).permute(0,3,1,2).float()
        y=F.max_pool2d(x,k['kernel_size'],k['stride'],k['padding'],k['dilation'],k['ceil_mode'])
        return self.typecast(Tensor(y.permute(0,2,3,1).reshape(1,1,-1,c),k['dtype']),k['dtype'])
    def global_avg_pool2d(self,x,**k):
        return self.typecast(Tensor(x.value.float().mean((1,2),keepdim=True),k['dtype']),k['dtype'])
    def add(self,x,y,**k):
        return Tensor(x.value+y.value,x.dtype)
    def relu(self,x):
        return Tensor(F.relu(x.value),x.dtype)
    def linear(self,x,w,bias=None,**k):
        y=x.value.float() @ w.value.float()
        if bias is not None:
            y=y+bias.value.float()
        return self.typecast(Tensor(y,k['dtype']),k['dtype'])
