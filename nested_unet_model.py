###############################################################################
## encoder : kernel size = 4, stride = 2, leakyReLU(0.2)
from keras import layers, Input, Model

###############################################################################
## generator encoder
class EncodeBlock(layers.Layer):
    def __init__(self, n_filters, use_bn=True):
        super(EncodeBlock, self).__init__()
        self.use_bn = use_bn       
        self.conv = layers.Conv2D(n_filters, 4, 2, "same", use_bias=False)
        self.batchnorm = layers.BatchNormalization()
        self.lrelu = layers.LeakyReLU(0.2)

    def call(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.batchnorm(x)
        return self.lrelu(x)
###############################################################################

###############################################################################
# ## generator decoder    
class DecodeBlock(layers.Layer):
    def __init__(self, f, dropout=True):
        super(DecodeBlock, self).__init__()
        self.dropout = dropout
        self.Transconv = layers.Conv2DTranspose(f, 4, 2, "same", use_bias=False)
        self.batchnorm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        
    def call(self, x):
        x = self.Transconv(x)
        x = self.batchnorm(x)
        if self.dropout:
            x = layers.Dropout(.5)(x)
        return self.relu(x)
###############################################################################

###############################################################################
## nested unet generator = ct data = (512,512,1)
class nUNetGenerator(Model):
    def __init__(self):
        super(nUNetGenerator, self).__init__()
        self.enc64 = EncodeBlock(64, use_bn=False)
        self.enc128 = EncodeBlock(128)
        self.enc256 = EncodeBlock(256)
        self.enc512_1 = EncodeBlock(512)
        self.enc512_2 = EncodeBlock(512)

        self.dec64_1 = DecodeBlock(64)
        self.dec64_2 = DecodeBlock(64)
        self.dec64_3 = DecodeBlock(64)
        self.dec64_4 = DecodeBlock(64, dropout=False)
        self.dec128_1 = DecodeBlock(128)
        self.dec128_2 = DecodeBlock(128)
        self.dec128_3 = DecodeBlock(128, dropout=False)
        self.dec256_1 = DecodeBlock(256)
        self.dec256_2 = DecodeBlock(256, dropout=False)
        self.dec512 = DecodeBlock(512)

        self.concat = layers.Concatenate()
        self.last_conv = layers.Conv2DTranspose(1, 4, 2, "same", use_bias=False)
    
    def call(self, x):
        x00 = self.enc64(x)  #256
        x10 = self.enc128(x00)  #128
        x20 = self.enc256(x10)  #64
        x30 = self.enc512_1(x20)  #32

        x40 = self.enc512_2(x30)  #16

        x10up = self.dec64_1(x10)
        x01 = self.concat([x00, x10up])
        x20up = self.dec128_1(x20)
        x11 = self.concat([x10, x20up])
        x30up = self.dec256_1(x30)
        x21 = self.concat([x20, x30up])

        ## middle-2
        x11up = self.dec64_2(x11)
        x02 = self.concat([x00, x01, x11up])
        x21up = self.dec128_2(x21)
        x12 = self.concat([x10, x11, x21up])

        ## middle-3
        x12up = self.dec64_3(x12)
        x03 = self.concat([x00, x01, x02, x12up])

        ## up
        x40up = self.dec512(x40)
        x31 = self.concat([x30, x40up])
        x31up = self.dec256_2(x31)
        x22 = self.concat([x20, x21, x31up])
        x22up = self.dec128_3(x22)
        x13 = self.concat([x10, x11, x12, x22up])
        x13up = self.dec64_4(x13)
        x04 = self.concat([x00, x01, x02, x03, x13up])

        x = self.last_conv(x04)
        return x
                
    def get_summary(self, input_shape=(512,512,1)):
        inputs = Input(input_shape)
        return Model(inputs, self.call(inputs)).summary()
###############################################################################

###############################################################################
## discriminator block
class DiscBlock(layers.Layer):
    def __init__(self, n_filters, stride=2, custom_pad=False, use_bn=True, act=True):
        super(DiscBlock, self).__init__()
        self.custom_pad = custom_pad
        self.use_bn = use_bn
        self.act = act
        
        if custom_pad:
            self.padding = layers.ZeroPadding2D()
            self.conv = layers.Conv2D(n_filters, 4, stride, "valid", use_bias=False)
        else:
            self.conv = layers.Conv2D(n_filters, 4, stride, "same", use_bias=False)
        
        self.batchnorm = layers.BatchNormalization() if use_bn else None
        self.lrelu = layers.LeakyReLU(0.2) if act else None
        
    def call(self, x):
        if self.custom_pad:
            x = self.padding(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
                
        if self.use_bn:
            x = self.batchnorm(x)
            
        if self.act:
            x = self.lrelu(x)
        return x 
###############################################################################

###############################################################################
## discriminator
class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.block1 = layers.Concatenate()
        self.block2 = DiscBlock(n_filters=64, stride=2, custom_pad=False, use_bn=False, act=True)
        self.block3 = DiscBlock(n_filters=128, stride=2, custom_pad=False, use_bn=True, act=True)
        self.block4 = DiscBlock(n_filters=256, stride=2, custom_pad=False, use_bn=True, act=True)
        self.block5 = DiscBlock(n_filters=512, stride=2, custom_pad=True, use_bn=True, act=True)
        self.block6 = DiscBlock(n_filters=1, stride=2, custom_pad=True, use_bn=False, act=False)
        self.sigmoid = layers.Activation("sigmoid")
        
        # filters = [64,128,256,512,1]
        # self.blocks = [layers.Concatenate()]
        # for i, f in enumerate(filters):
        #     self.blocks.append(DiscBlock(
        #         n_filters=f,
        #         strides=2 if i<3 else 1,
        #         custom_pad=False if i<3 else True,
        #         use_bn=False if i==0 and i==4 else True,
        #         act=True if i<4 else False
        #     ))
    
    def call(self, x, y):
        out = self.block1([x, y])
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        return self.sigmoid(out)
    
    def get_summary(self, x_shape=(512,512,1), y_shape=(512,512,1)):
        x, y = Input(x_shape), Input(y_shape) 
        return Model((x, y), self.call(x, y)).summary()
###############################################################################
